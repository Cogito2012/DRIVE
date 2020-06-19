import os, cv2
import numpy as np
from tqdm import tqdm

def get_data_list(root_path, phase):
    # video path
    rgb_path = os.path.join(root_path, phase, 'rgb')
    assert os.path.exists(rgb_path)
    # loop for each type of accident
    data_list = []
    for accident in sorted(os.listdir(rgb_path)):
        accident_rgb_path = os.path.join(rgb_path, accident)
        for vid in sorted(os.listdir(accident_rgb_path)):
            data_list.append(accident + '/' + vid)
    return data_list


def read_video_frames(video_path):
        """ Read video frames
        """
        assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
        frame_ids, video_data = [], []
        for filename in sorted(os.listdir(video_path)):  # we must sort the image files to ensure a sequential frame order
            # get frame id list
            fid = int(filename.split('.')[0])
            frame_ids.append(fid)
            # read video frame
            im = cv2.imread(os.path.join(video_path, filename))
            video_data.append(im)
        return video_data, frame_ids


def read_focus_images(frame_ids, focus_path):
    """ Read focus images
    """
    assert os.path.exists(focus_path), "Path does not exist: %s"%(focus_path)
    focus_data = []
    for fid in frame_ids:
        focus_file = os.path.join(focus_path, "%04d.png"%(fid))
        if not os.path.exists(focus_file):
            focus_file = os.path.join(focus_path, "%04d.jpg"%(fid))
            if not os.path.exists(focus_file):
                print('Focus file does not exist: %s'%(focus_file))
                break
        focus = cv2.imread(focus_file)  # BGR: (660, 1584, 3)
        focus_data.append(focus)
    return focus_data

    
if __name__ == "__main__":

    root_path = './data/DADA-2000'
    fps = 30

    for phase in ['training', 'testing', 'validation']:
        # get the data list
        data_list = get_data_list(root_path, phase)

        for index, dataID in tqdm(enumerate(data_list), desc=phase, total=len(data_list)):
            # video path
            video_path = os.path.join(root_path, phase, 'rgb', data_list[index])
            # focus path
            focus_path = os.path.join(root_path, phase, 'focus', data_list[index])

            # result video path
            result_video_path = os.path.join(root_path, phase, 'rgb_videos', data_list[index].split('/')[0])
            if not os.path.exists(result_video_path):
                os.makedirs(result_video_path)
            video_file = os.path.join(result_video_path, data_list[index].split('/')[1] + '.avi')
            # result focus path
            result_focus_path = os.path.join(root_path, phase, 'focus_videos', data_list[index].split('/')[0])
            if not os.path.exists(result_focus_path):
                os.makedirs(result_focus_path)
            focus_file = os.path.join(result_focus_path, data_list[index].split('/')[1] + '.avi')

            if os.path.exists(video_file) and os.path.exists(focus_file):
                continue

            # read video frame data
            video_data, frame_ids = read_video_frames(video_path)
            assert len(video_data) > 0, "Empty video data!"
            # read focus data
            focus_data = read_focus_images(frame_ids, focus_path)
            assert len(focus_data) > 0, "Empty focus data!"

            # write video data
            video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (video_data[0].shape[1], video_data[0].shape[0]))
            for t, frame in enumerate(video_data):
                video_writer.write(frame)
            # write focus video
            focus_writer = cv2.VideoWriter(focus_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (focus_data[0].shape[1], focus_data[0].shape[0]))
            for t, frame in enumerate(focus_data):
                focus_writer.write(frame)
