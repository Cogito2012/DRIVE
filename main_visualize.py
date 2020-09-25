import os, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_frames_from_videos(root_path, vid_name, start, end, phase='testing', interval=1):
    """Read video frames
    """
    video_path = os.path.join(root_path, phase, 'rgb_videos', vid_name + '.avi')
    assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
    # get the video data
    cap = cv2.VideoCapture(video_path)
    video_data = []
    for fid in range(start, end + 1, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        assert ret, "read video failed! file: %s frame: %d"%(video_path, fid)
        video_data.append(frame)
    return video_data

def create_curve_video(pred_scores, toa, n_frames, frame_interval):
    # background
    fig, ax = plt.subplots(1, figsize=(30,5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames*frame_interval + 1, frame_interval*2), fontsize=fontsize)
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
            plt.plot(xvals[:(t+1)], pred_scores[:(t+1)], linewidth=3.0, color='r')
            plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
            if toa >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, n_frames * frame_interval]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            curve_writer.grab_frame()
    # read frames
    cap = cv2.VideoCapture("tmp_curve_video.mp4")
    ret, frame = cap.read()
    curve_frames = []
    while (ret):
        curve_frames.append(frame)
        ret, frame = cap.read()
    return curve_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Results')
    # For training and testing
    parser.add_argument('--data_path', default="data/DADA-2000-small",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--test_results', default='output/SAC_MLNet_GG/eval/results.npz',
                        help='Result file of testing data.')
    parser.add_argument('--output', default='./output/SAC_MLNet_GG',
                        help='Directory of the output. ')
    args = parser.parse_args()
    frame_interval = 5
    image_size = [330, 792]
    height, width = 480, 640

    if not os.path.exists(args.test_results):
        print('Results file not found!')
        os.sys.exit()
    save_dict = np.load(args.test_results, allow_pickle=True)
    all_pred_scores, all_gt_labels, all_pred_masks, all_gt_masks, all_toas, all_vids = \
            save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_masks'], save_dict['gt_masks'], save_dict['toas'], save_dict['vids']

    # prepare output directory
    output_dir = os.path.join(args.output, 'vis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visits = []
    for i, vids in enumerate(all_vids):
        accid, vid, start, end = vids.tolist()
        vidname = '%d/%03d'%(accid, vid)
        if vidname in visits:
            continue    
        if len(visits) > 19:
            break
        pred_scores = all_pred_scores[i]
        gt_labels = all_gt_labels[i]
        pred_masks = all_pred_masks[i]
        gt_masks = all_gt_masks[i]
        toa = int(all_toas[i] * 30)
        if not gt_labels > 0:
            continue
        visits.append(vidname)

        print("accident ID=%d, video ID=%d"%(accid, vid))
        
        frames = read_frames_from_videos(args.data_path, vidname, start, end, phase='testing', interval=5)

        curve_frames = create_curve_video(pred_scores, toa, len(frames), frame_interval)

        vis_file = os.path.join(output_dir, 'vis_%d_%03d.avi'%(accid, vid))
        video_writer = cv2.VideoWriter(vis_file, cv2.VideoWriter_fourcc(*'DIVX'), 2.0, (image_size[1], image_size[0]))

        for t, frame in enumerate(frames):
            # add pred_mask as heatmap
            mask = cv2.resize(pred_masks[t], (image_size[1], image_size[0]), interpolation = cv2.INTER_AREA)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask + 1e-6))  # normalize
            heatmap = np.uint8(255 * mask)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
            # add curve
            curve_img = curve_frames[t]
            curve_height = int(curve_img.shape[0] * (image_size[1] / curve_img.shape[1]))
            curve_img = cv2.resize(curve_img, (image_size[1], curve_height), interpolation = cv2.INTER_AREA)
            frame[image_size[0]-curve_height:image_size[0]] = cv2.addWeighted(frame[image_size[0]-curve_height:image_size[0]], 0.5, curve_img, 0.5, 0)
            video_writer.write(frame)
        
