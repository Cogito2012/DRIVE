import os, cv2, shutil
import numpy as np
import argparse

def read_coords(coord_file):
    coord_data, inds_pos = [], []
    assert os.path.exists(coord_file), "File does not exist! %s"%coord_file
    with open(coord_file, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            x_coord = int(line.strip().split(',')[0])
            y_coord = int(line.strip().split(',')[1])
            if x_coord > 0 and y_coord > 0:
                inds_pos.append(ind)
            coord_data.append([x_coord, y_coord])
    coord_data = np.array(coord_data, dtype=np.int32)
    inds_pos = np.array(inds_pos, dtype=np.int32)
    return coord_data, inds_pos

def write_coords(coord_data, frame_ids, coord_file):
    coord_dir = os.path.dirname(coord_file)
    if not os.path.exists(coord_dir):
        os.makedirs(coord_dir)
    with open(coord_file, 'w') as f:
        for i in frame_ids:
            x_coord = int(coord_data[i, 0])
            y_coord = int(coord_data[i, 1])
            f.writelines('%d,%d\n'%(x_coord, y_coord))

def reduce_video(src_file, dst_file, ratio, frame_ids):
    assert os.path.exists(src_file), "File does not exist! %s"%src_file
    video_dir = os.path.dirname(dst_file)
    if os.path.exists(dst_file):
        return
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    # video capture of src video
    cap_src = cv2.VideoCapture(src_file)
    ret, frame = cap_src.read()
    ind = 0
    # dest capture
    dst_size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))  # (width, height)
    cap_dst = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*'XVID'), 30, dst_size)
    while (ret):
        if ind in frame_ids:
            frame_resize = cv2.resize(frame, dst_size)
            cap_dst.write(frame_resize)
        # read next frame
        ret, frame = cap_src.read()
        ind += 1
    

def reduce_data(data_path, ratio, max_frames, subset, result_path):
    # the input path
    coord_path_src = os.path.join(data_path, subset, 'coordinate')
    focus_path_src = os.path.join(data_path, subset, 'focus_videos')
    salmap_path_src = os.path.join(data_path, subset, 'salmap_videos')
    video_path_src = os.path.join(data_path, subset, 'rgb_videos')
    # the input path
    coord_path_dst = os.path.join(result_path, subset, 'coordinate')
    focus_path_dst = os.path.join(result_path, subset, 'focus_videos')
    salmap_path_dst = os.path.join(result_path, subset, 'salmap_videos')
    video_path_dst = os.path.join(result_path, subset, 'rgb_videos')
    
    for accID in sorted(os.listdir(coord_path_src)):
        txtfile_dir = os.path.join(coord_path_src, accID)
        for filename in sorted(os.listdir(txtfile_dir)):
            coord_file_src = os.path.join(txtfile_dir, filename)
            coord_data, inds_pos = read_coords(coord_file_src)
            if inds_pos.shape[0] == 0:
                continue  # ignore videos without any accident

            # remove the frames after accident ends
            video_end = min(inds_pos[-1] + 1 + 16, coord_data.shape[0])
            video_start = max(0, video_end - max_frames)
            frame_ids = np.arange(video_start, video_end)

            vid = filename.split('_')[0]
            print("Processing the video: %s/%s, # frames: %d"%(accID, vid, len(frame_ids)))
            # resize & write coords
            coord_file_dst = os.path.join(coord_path_dst, accID, filename)
            write_coords(ratio * coord_data, frame_ids, coord_file_dst)

            # read focus videos
            focus_video_src = os.path.join(focus_path_src, accID, vid + '.avi')
            focus_video_dst = os.path.join(focus_path_dst, accID, vid + '.avi')
            reduce_video(focus_video_src, focus_video_dst, ratio, frame_ids)

            # read salmap videos
            salmap_video_src = os.path.join(salmap_path_src, accID, vid + '.avi')
            salmap_video_dst = os.path.join(salmap_path_dst, accID, vid + '.avi')
            reduce_video(salmap_video_src, salmap_video_dst, ratio, frame_ids)

            # read rgb videos
            rgb_video_src = os.path.join(video_path_src, accID, vid + '.avi')
            rgb_video_dst = os.path.join(video_path_dst, accID, vid + '.avi')
            reduce_video(rgb_video_src, rgb_video_dst, ratio, frame_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reduce the size of DADA-2000')
    parser.add_argument('--data_path', default="./DADA-2000",
                        help='Directory to the original DADA-2000 folder.')
    parser.add_argument('--result_path', default="./DADA-2000-small",
                        help='Directory to the result DADA-2000 folder.')
    args = parser.parse_args()

    ratio = 0.5
    max_frames = 450  # for fps=30, the maxtime=20 s after clipped
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    reduce_data(args.data_path, ratio, max_frames, 'training', args.result_path)
    reduce_data(args.data_path, ratio, max_frames, 'testing', args.result_path)
    reduce_data(args.data_path, ratio, max_frames, 'validation', args.result_path)
