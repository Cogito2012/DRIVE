import os, cv2, shutil
import numpy as np

def read_coords(coord_file):
    coord_data = []
    assert os.path.exists(coord_file), "File does not exist! %s"%coord_file
    with open(coord_file, 'r') as f:
        for line in f.readlines():
            x_coord = int(line.strip().split(',')[0])
            y_coord = int(line.strip().split(',')[1])
            coord_data.append([x_coord, y_coord])
    coord_data = np.array(coord_data, dtype=np.int32)
    return coord_data

def write_coords(coord_data, coord_file):
    coord_dir = os.path.dirname(coord_file)
    if not os.path.exists(coord_dir):
        os.makedirs(coord_dir)
    with open(coord_file, 'w') as f:
        for coord in coord_data:
            x_coord = int(coord[0])
            y_coord = int(coord[1])
            f.writelines('%d,%d\n'%(x_coord, y_coord))

def resize_video(src_file, dst_file, ratio):
    assert os.path.exists(src_file), "File does not exist! %s"%src_file
    video_dir = os.path.dirname(dst_file)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    # video capture of src video
    cap_src = cv2.VideoCapture(src_file)
    ret, frame = cap_src.read()
    # dest capture
    dst_size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))  # (width, height)
    cap_dst = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*'XVID'), 30, dst_size)
    while (ret):
        frame_resize = cv2.resize(frame, dst_size)
        cap_dst.write(frame_resize)
        # read next frame
        ret, frame = cap_src.read()
    

def resize_data(data_path, ratio, subset, result_path):
    # the input path
    coord_path_src = os.path.join(data_path, subset, 'coordinate')
    focus_path_src = os.path.join(data_path, subset, 'focus_videos')
    video_path_src = os.path.join(data_path, subset, 'rgb_videos')
    # the input path
    coord_path_dst = os.path.join(result_path, subset, 'coordinate')
    focus_path_dst = os.path.join(result_path, subset, 'focus_videos')
    video_path_dst = os.path.join(result_path, subset, 'rgb_videos')
    
    for accID in sorted(os.listdir(coord_path_src)):
        txtfile_dir = os.path.join(coord_path_src, accID)
        for filename in sorted(os.listdir(txtfile_dir)):
            vid = filename.split('_')[0]
            print("Processing the video: %s/%s"%(accID, vid))
            # read the coordinates
            coord_file_src = os.path.join(txtfile_dir, filename)
            coord_data = read_coords(coord_file_src)
            # resize & write coords
            coord_file_dst = os.path.join(coord_path_dst, accID, filename)
            write_coords(ratio * coord_data, coord_file_dst)

            # read focus videos
            focus_video_src = os.path.join(focus_path_src, accID, vid + '.avi')
            focus_video_dst = os.path.join(focus_path_dst, accID, vid + '.avi')
            resize_video(focus_video_src, focus_video_dst, ratio)
            # read rgb videos
            rgb_video_src = os.path.join(video_path_src, accID, vid + '.avi')
            rgb_video_dst = os.path.join(video_path_dst, accID, vid + '.avi')
            resize_video(rgb_video_src, rgb_video_dst, ratio)


if __name__ == "__main__":
    data_path = './DADA-2000'
    ratio = 0.5
    result_path = './DADA-2000-small'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    resize_data(data_path, ratio, 'training', result_path)
    resize_data(data_path, ratio, 'testing', result_path)
    resize_data(data_path, ratio, 'validation', result_path)
