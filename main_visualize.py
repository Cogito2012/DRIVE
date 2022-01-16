import os, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.saliency.mlnet import MLNet
import torch
from torchvision import transforms
from src.data_transform import ProcessImages, ProcessFixations
from src.TorchFovea import TorchFovea


def read_frames_from_videos(root_path, vid_name, start, end, folder, phase='testing', interval=1):
    """Read video frames
    """
    video_path = os.path.join(root_path, phase, folder, vid_name + '.avi')
    assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
    # get the video data
    cap = cv2.VideoCapture(video_path)
    video_data = []
    for fid in range(start, end + 1, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        assert ret, "read video failed! file: %s frame: %d"%(video_path, fid)
        video_data.append(frame)
    video_data = np.array(video_data)
    return video_data


def create_curve_video(pred_scores, toa, n_frames, frame_interval):
    # background
    fig, ax = plt.subplots(1, figsize=(30,5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames*frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig('tmp_curve.png')
    # draw curves
    from matplotlib.animation import FFMpegWriter
    curve_writer = FFMpegWriter(fps=2, metadata=dict(title='Movie Test', artist='Matplotlib',comment='Movie support!'))
    with curve_writer.saving(fig, "tmp_curve_video.mp4", 100):
        xvals = np.arange(n_frames+1) * frame_interval
        pred_scores = pred_scores.tolist() + [pred_scores[-1]]
        for t in range(1, n_frames+1):
            plt.plot(xvals[:(t+1)], pred_scores[:(t+1)], linewidth=5.0, color='r')
            plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
            if toa >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, n_frames * frame_interval]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            curve_writer.grab_frame()
    plt.close()
    # read frames
    cap = cv2.VideoCapture("tmp_curve_video.mp4")
    ret, frame = cap.read()
    curve_frames = []
    while (ret):
        curve_frames.append(frame)
        ret, frame = cap.read()
    return curve_frames


def plot_scores(pred_scores, toa, n_frames, frame_interval, out_file):
    # background
    fig, ax = plt.subplots(1, figsize=(30,5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)

    xvals = np.arange(n_frames) * frame_interval
    plt.plot(xvals, pred_scores, linewidth=5.0, color='r')
    plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
    if toa >= 0:
        plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        x = [toa, n_frames * frame_interval]
        y1 = [0, 0]
        y2 = [1, 1]
        ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)

    # plt.ylabel('Probability', fontsize=fontsize)
    # plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames*frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    


def minmax_norm(salmap):
    """Normalize the saliency map with min-max
    salmap: (B, 1, H, W)
    """
    batch_size, height, width = salmap.size(0), salmap.size(2), salmap.size(3)
    salmap_data = salmap.view(batch_size, -1)  # (B, H*W)
    min_vals = salmap_data.min(1, keepdim=True)[0]  # (B, 1)
    max_vals = salmap_data.max(1, keepdim=True)[0]  # (B, 1)
    salmap_norm = (salmap_data - min_vals) / (max_vals - min_vals + 1e-6)
    salmap_norm = salmap_norm.view(batch_size, 1, height, width)
    return salmap_norm
    

def generate_attention(frame_data, data_trans, salmodel, fovealmodel, fixations, image_size, n_slice=1, rho_list=None, device=None):
    assert frame_data.shape[0] % n_slice == 0, "invalid n_slice!"
    slice_size = int(frame_data.shape[0] / n_slice)
    attention_maps = []
    for i in range(n_slice):
        # get bottom-up attention
        input_data = torch.FloatTensor(data_trans(frame_data[i*slice_size:(i+1)*slice_size])).to(device)
        fixation_data = torch.from_numpy(fixations[i*slice_size:(i+1)*slice_size]).to(device)
        foveal_data = fovealmodel.foveate(input_data, fixation_data)
        with torch.no_grad():
            saliency_bu = salmodel(input_data)
            saliency_bu = minmax_norm(saliency_bu)
            saliency_td = salmodel(foveal_data)
            saliency_td = minmax_norm(saliency_td)
        saliency_bu = saliency_bu.squeeze(1).cpu().numpy()
        saliency_td = saliency_td.squeeze(1).cpu().numpy()
        rho = np.expand_dims(np.expand_dims(np.array(rho_list[i*slice_size:(i+1)*slice_size]), axis=1), axis=2)
        saliency = (1 - rho) * saliency_bu + rho * saliency_td
        # padd the saliency maps to image size
        salmap = saliency_padding(saliency, image_size)
        attention_maps.append(salmap)
    attention_maps = np.concatenate(attention_maps, axis=0)
    return attention_maps


def saliency_padding(saliency, image_size):
    """Up padding the saliency (B, 60, 80) to image size (B, 330, 792)
    """
    # get size and ratios
    height, width = saliency.shape[1:]
    rows_rate = image_size[0] / height  # h ratio (5.5)
    cols_rate = image_size[1] / width   # w ratio (9.9)
    # padding
    if rows_rate > cols_rate:
        pass
    else:
        new_rows = (image_size[0] * width) // image_size[1]
        patch_ctr = saliency[:, ((height - new_rows) // 2):((height - new_rows) // 2 + new_rows), :]
        patch_ctr = np.rollaxis(patch_ctr, 0, 3)
        padded = cv2.resize(patch_ctr, (image_size[1], image_size[0]))
        padded = np.rollaxis(padded, 2, 0)
    return padded


def fixation_padding(fixation, height, width, image_size):
    """Up padding the fixations (B, 2) defined in (height, width)=(480, 640) to image size (330, 792)
    """
    # get size and ratios
    rows_rate = image_size[0] / height  # h ratio (5.5)
    cols_rate = image_size[1] / width   # w ratio (9.9)
    # padding
    if rows_rate > cols_rate:
        pass
    else:
        new_rows = (image_size[0] * width) // image_size[1]
        fixation_shifted[1] = fixation[1] - (height - new_rows) // 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Results')
    # For training and testing
    parser.add_argument('--data_path', default="data/DADA-2000-small",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--sal_ckpt', default='models/saliency/mlnet_25.pth',
                        help='Pretrained model for bottom-up saliency prediciton.')
    parser.add_argument('--test_results', default='output/DADA2KS_Full_SACAE_Final/eval/results.npz',
                        help='Result file of testing data.')
    parser.add_argument('--rho', type=float, default=0.5,
                        help='The rho value')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='The margin value')
    parser.add_argument('--static', action='store_true',
                        help='whether to use static fusion test')
    parser.add_argument('--no_curve_overlap', action='store_true',
                        help='whether to overlap curve figure on video.')
    parser.add_argument('--gt_compare', action='store_true',
                        help='whether to compare with GT saliency video')
    parser.add_argument('--output', default='./output/DADA2KS_Full_SACAE_Final/vis_results-01152022',
                        help='Directory of the output. ')
    args = parser.parse_args()
    frame_interval = 5
    display_fps = 2
    image_size = [330, 792]
    height, width = 480, 640

    # environmental model
    device = torch.device("cuda")
    observe_model = MLNet((height, width))
    assert os.path.exists(args.sal_ckpt), "Checkpoint directory does not exist! %s"%(args.sal_ckpt)
    ckpt = torch.load(args.sal_ckpt, map_location=device)
    observe_model.load_state_dict(ckpt['model'])
    observe_model.to(device)
    observe_model.eval()
    fovealmodel = TorchFovea((height, width), min(height, width)/6.0, level=5, factor=2, device=device)
    # transform
    data_trans = transforms.Compose([ProcessImages((height, width), mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])])

    if not os.path.exists(args.test_results):
        print('Results file not found!')
        os.sys.exit()
    save_dict = np.load(args.test_results, allow_pickle=True)
    all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = \
            save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_fixations'], save_dict['gt_fixations'], save_dict['toas'], save_dict['vids']

    # prepare output directory
    output_folder = str(int(args.rho * 100)) if args.static else 'dynamic'
    output_dir = os.path.join(args.output, output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_gt_dir = os.path.join(args.output, 'ground_truth')
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)
    
    target_list = ['1/022', '6/058', '11/107', '11/113', '14/013', '38/039', '39/023']
    # target_list = ['1/022']
    for i, vids in enumerate(all_vids):
        accid, vid, start, end = vids.tolist()
        vidname = '%d/%03d'%(accid, vid)
        if vidname not in target_list:
            continue
        pred_scores = all_pred_scores[i]
        gt_labels = all_gt_labels[i]
        pred_fixations = all_pred_fixations[i]
        gt_fixations = all_gt_fixations[i]
        toa = int(all_toas[i] * 30)
        if not gt_labels > 0:
            continue

        print("accident ID=%d, video ID=%d"%(accid, vid))
        
        # read frames
        frames = read_frames_from_videos(args.data_path, vidname, start, end, 'rgb_videos', phase='testing', interval=frame_interval)
        if not args.no_curve_overlap:
            # create curves
            curve_frames = create_curve_video(pred_scores, toa, len(frames), frame_interval)
        # plot curves
        curve_dir = os.path.join(args.output, 'curves')
        if not os.path.exists(curve_dir):
            os.makedirs(curve_dir)
        out_file = os.path.join(curve_dir, 'curve_%d_%d.png'%(accid, vid))
        curve = plot_scores(pred_scores, toa, len(frames), frame_interval, out_file)

        # get saliency maps
        if not args.static:
            rho = np.minimum(pred_scores, args.margin)
        else:
            rho = [args.rho] * len(pred_scores)
        attention_maps = generate_attention(frames, data_trans, observe_model, fovealmodel, pred_fixations, image_size, n_slice=5, rho_list=rho, device=device)

        gt_salmaps = read_frames_from_videos(args.data_path, vidname, start, end, 'salmap_videos', phase='testing', interval=frame_interval)

        vis_file = os.path.join(output_dir, 'vis_%d_%03d.avi'%(accid, vid))
        height_vis = image_size[0] if not args.gt_compare else image_size[0] * 2
        video_writer = cv2.VideoWriter(vis_file, cv2.VideoWriter_fourcc(*'DIVX'), display_fps, (image_size[1], height_vis))
        for t, frame in enumerate(frames):
            # add pred_mask as heatmap
            heatmap = cv2.applyColorMap((attention_maps[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame_vis = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            # add curve
            if not args.no_curve_overlap:
                curve_img = curve_frames[t]
                curve_height = int(curve_img.shape[0] * (image_size[1] / curve_img.shape[1]))
                curve_img = cv2.resize(curve_img, (image_size[1], curve_height), interpolation = cv2.INTER_AREA)
                frame_vis[image_size[0]-curve_height:image_size[0]] = cv2.addWeighted(frame_vis[image_size[0]-curve_height:image_size[0]], 0.3, curve_img, 0.7, 0)
            if args.gt_compare:
                # add gt_mask as heatmap
                gt_salmap = cv2.applyColorMap((gt_salmaps[t]), cv2.COLORMAP_JET)
                frame_vis_gt = cv2.addWeighted(frame, 0.5, gt_salmap, 0.5, 0)
                # add text
                cv2.putText(frame_vis, 'Prediction', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_vis_gt, 'Ground Truth', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                frame_vis = np.concatenate((frame_vis_gt, frame_vis), axis=0)
            video_writer.write(frame_vis)

        if not args.gt_compare:
            # generate GT saliency video
            vis_file = os.path.join(output_gt_dir, 'vis_%d_%03d.avi'%(accid, vid))
            video_writer = cv2.VideoWriter(vis_file, cv2.VideoWriter_fourcc(*'DIVX'), display_fps, (image_size[1], image_size[0]))
            for t, frame in enumerate(frames):
                # add gt_mask as heatmap
                heatmap = cv2.applyColorMap((gt_salmaps[t]), cv2.COLORMAP_JET)
                frame_vis = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
                video_writer.write(frame_vis)
        
