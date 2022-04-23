import os, cv2
import numpy as np

def get_video_frames(videofile):
    assert os.path.exists(videofile), "File does not exist: %s"%(videofile)
    # get the video data
    cap = cv2.VideoCapture(videofile)
    ret, frame = cap.read()
    video_data = []
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
    
    return video_data

def read_coord_arrays(coord_file):
    """ Read coordinate array
    """
    assert os.path.exists(coord_file), "File does not exist: %s"%(coord_file)
    coord_data = []
    with open(coord_file, 'r') as f:
        for line in f.readlines():
            x_coord = int(line.strip().split(',')[0])
            y_coord = int(line.strip().split(',')[1])
            coord_data.append([x_coord, y_coord])
    coord_data = np.array(coord_data, dtype=np.int32)
    return coord_data

def read_mapping(map_file):
    map_dict = {'ID_data':[], 'ID_paper':[], 'participants':[], 'accident':[]}
    with open(map_file, 'r') as f:
        for line in f.readlines():
            strs = line.strip().split(',')
            map_dict['ID_data'].append(int(strs[0]))
            map_dict['ID_paper'].append(int(strs[1]))
            obj_list = strs[2:4]
            if 'self' in obj_list:
                obj_list.remove('self')
            map_dict['participants'].append(obj_list)
            map_dict['accident'].append(strs[4])
    return map_dict

if __name__ == "__main__":

    root_path = './data/DADA-2000'
    fps = 30
    phase = 'testing'
    atype = '40'  # '1', '16', '11', '34', '40'
    sequence = '080'  # '022', '001', '097', '088', '080'
    barWidth = 60
    vis_video_file = './vis_data/vis_' + atype + '_' + sequence + '.avi'

    video_file = os.path.join(root_path, phase, 'rgb_videos', atype, sequence + '.avi')
    salmap_file = os.path.join(root_path, phase, 'salmap_videos', atype, sequence + '.avi')
    coord_file = os.path.join(root_path, phase, 'coordinate', atype, sequence + '_coordinate.txt')
    mapping_file = os.path.join(root_path, 'mapping.txt')
    
    video_data = get_video_frames(video_file)
    salmap_data = get_video_frames(salmap_file)
    assert len(video_data) == len(salmap_data)

    coord_data = read_coord_arrays(coord_file)
    assert len(video_data) == coord_data.shape[0]

    map_dicts = read_mapping(mapping_file)

    video_writer = cv2.VideoWriter(vis_video_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (video_data[0].shape[1], video_data[0].shape[0]))
    heatmap = np.zeros_like(video_data[0])
    h, w, c = heatmap.shape
    progress_bar = np.full((barWidth, w, c), (255, 255, 0), np.uint8)  # cyan color
    for t, (frame, salmap, fixation) in enumerate(zip(video_data, salmap_data, coord_data)):
        # add saliency heatmap as overlap
        heatmap = cv2.applyColorMap(salmap, cv2.COLORMAP_JET)
        visframe = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        # add colorbar for temporal axis
        step = int(np.ceil(frame.shape[1] / len(video_data)))
        progress_bar[:, 0:t*step, 0] = 0  # fill green color (0, 255, 0)

        if fixation[0] != 0 and fixation[1] != 0:
            # add fixation point
            visframe = cv2.drawMarker(visframe, tuple(fixation), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=5)
            # add temporal annotations
            progress_bar[:, t*step: min((t+1)*step, w), 0] = 0  # fill red color (0, 0, 255)
            progress_bar[:, t*step: min((t+1)*step, w), 1] = 0
            progress_bar[:, t*step: min((t+1)*step, w), 2] = 255
            # add textual description
            idx = map_dicts['ID_data'].index(int(atype))
            participants = map_dicts['participants'][idx]
            tag1 = 'participants: %s'%(participants[0])
            if len(participants) > 1:
                tag1 += ',' + participants[1]
            tag2 = 'accidents: %s'%(map_dicts['accident'][idx])
            cv2.putText(visframe, tag1, (60, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(visframe, tag2, (60, 120), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,255), 2, cv2.LINE_AA)

        visframe[frame.shape[0]-barWidth-1: frame.shape[0]-1, :, :] = cv2.addWeighted(visframe[frame.shape[0]-barWidth-1: frame.shape[0]-1, :, :], 0.3, progress_bar, 0.7, 0)
        # write result video
        video_writer.write(visframe)