import os, cv2
import argparse
import numpy as np


def select_frames(frame_ids, num=-1):
    if num < 0:
        inds = np.arange(len(frame_ids))  # select all T frames
    else:
        inds = np.arange(min(num, len(frame_ids)))  # select top-N or all (N >= T)
    sel_frames = [frame_ids[i] for i in inds]
    return sel_frames, inds

def read_frames_from_videos(root_path, vid_name, phase='testing', interval=1, max_frames=-1):
    """Read video frames
    """
    video_path = os.path.join(root_path, phase, 'rgb_videos', vid_name + '.avi')
    assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
    frame_ids, video_data = [], []
    # get the video data
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    video_data = []
    fid = 1
    while (ret):
        if (fid-1) % interval == 0:  # starting from 1
            # get frame id list
            frame_ids.append(fid)
            # read video frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB: (660, 1584, 3)
            video_data.append(frame)
        ret, frame = cap.read()
        fid += 1
    if max_frames > 0:
        frame_ids, inds = select_frames(frame_ids, num=max_frames)
        video_data = [video_data[i] for i in inds]
    # video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
    return video_data, frame_ids


def point_to_scales(point, height, width, image_size):
        """Transform the point that is defined on [480, 640] plane into scales ranging from -1 to 1 on image plane
        point: [x, y]
        """
        point = point.copy()
        rows_rate = image_size[0] / height  # 660 / 240
        cols_rate = image_size[1] / width   # 1584 / 320
        if rows_rate > cols_rate:
            new_cols = (image_size[1] * height) // image_size[0]
            point[0] = point[0] - (width - new_cols) // 2
            scale_x = (point[0] - 0.5 * new_cols) / (0.5 * new_cols)
            scale_y = (0.5 * height - point[1]) / (0.5 * height)
        else:
            new_rows = (image_size[0] * width) // image_size[1]
            point[1] = point[1] - (height - new_rows) // 2
            scale_y = (0.5 * new_rows - point[1]) / (0.5 * new_rows)
            scale_x = (point[0] - 0.5 * width) / (0.5 * width)
        scales = np.array([scale_x, scale_y])
        return scales


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Results')
    # For training and testing
    parser.add_argument('--data_path', default="data/DADA-2000",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--test_results', default='output/SAC_AE_MLNet/eval/results.npy',
                        help='Result file of testing data.')
    parser.add_argument('--output', default='./output/SAC_AE_MLNet',
                        help='Directory of the output. ')
    args = parser.parse_args()
    frame_interval = 5
    image_size = [660, 1584]
    height, width = 480, 640

    if not os.path.exists(args.test_results):
        raise FileNotFoundError
        os.sys.exit()
    save_dict = np.load(args.test_results, allow_pickle=True)
    save_dict = save_dict.item()

    fix_preds = save_dict['fix_preds']
    fix_gts = save_dict['gt_fixes']
    vids = save_dict['vids']

    # prepare output directory
    output_dir = os.path.join(args.output, 'vis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, fix_pred in enumerate(fix_preds):
        if i > 19:
            break
        fix_gt = fix_gts[i]
        vid = vids[i][0]
        print("accident ID=%d, video ID=%d"%(vid[0], vid[1]))
        vidname = '%d/%03d'%(vid[0], vid[1])
        frames, _ = read_frames_from_videos(args.data_path, vidname, phase='testing', interval=5, max_frames=120)

        vis_file = os.path.join(output_dir, 'vis_%d_%03d.avi'%(vid[0], vid[1]))
        video_writer = cv2.VideoWriter(vis_file, cv2.VideoWriter_fourcc(*'DIVX'), 2.0, (image_size[1], image_size[0]))

        end_time = np.where(fix_gt[:, 0] > 0)[0][-1]
        for j, img in enumerate(frames):
            # if j > 0 and fix_gt[j][0] - fix_gt[j-1][0] < 0:
            #     break
            if j >= end_time:
                break
            radius = 40
            scales = point_to_scales(fix_pred[j], height, width, image_size)
            fix_pred_j = (int((scales[0] + 1.0) * 0.5 * image_size[1]), int((1.0 - scales[1]) * 0.5 * image_size[0]))  # (x, y)
            img = cv2.circle(img, fix_pred_j, radius, (0, 0, 255), -1)  # red circle
            if fix_gt[j][0] > 0 and fix_gt[j][1] > 0:
                scales = point_to_scales(fix_gt[j], height, width, image_size)
                fix_gt_j = (int((scales[0] + 1.0) * 0.5 * image_size[1]), int((1.0 - scales[1]) * 0.5 * image_size[0]))  # (x, y)
                img = cv2.circle(img, fix_gt_j, radius, (0, 255, 255), -1)  # yellow circle
            # save as video
            video_writer.write(img)
