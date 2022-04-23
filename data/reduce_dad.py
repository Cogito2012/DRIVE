import os, cv2
import numpy as np
import argparse


def resize_video(video_src, video_dst, ratio, n_frames):
    assert os.path.exists(video_src), "File does not exist! %s"%video_src
    cap_src = cv2.VideoCapture(video_src)
    ret, frame = cap_src.read()
    # dest capture
    dst_size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))  # (width, height)
    cap_dst = cv2.VideoWriter(video_dst, cv2.VideoWriter_fourcc(*'mp4v'), FPS, dst_size)
    num = 0
    while (ret):
        frame_resize = cv2.resize(frame, dst_size)
        cap_dst.write(frame_resize)
        # read next frame
        ret, frame = cap_src.read()
        num += 1
        if num >= n_frames:
            break
    if num < n_frames:
        for _ in range(n_frames - num):
            cap_dst.write(frame_resize)


def reduce_data(data_path, ratio, n_frames, result_path):

    for subset in ['training', 'testing']:
        subset_srcpath = os.path.join(data_path, 'videos', subset)
        subset_dstpath = os.path.join(result_path, subset)
        for tag in ['positive', 'negative']:
            videos_srcpath = os.path.join(subset_srcpath, tag)
            videos_dstpath = os.path.join(subset_dstpath, tag)
            if not os.path.exists(videos_dstpath):
                os.makedirs(videos_dstpath)
            # process each video
            for filename in sorted(os.listdir(videos_srcpath)):
                print("processing video: %s/%s/%s"%(subset, tag, filename))
                # output video
                video_dst = os.path.join(videos_dstpath, filename)
                if os.path.exists(video_dst):
                    continue
                # input video
                video_src = os.path.join(videos_srcpath, filename)
                # process
                resize_video(video_src, video_dst, ratio, n_frames)


def write_listfile(files_set, data_path, features_path, list_file, subset='training'):
    with open(list_file, 'w') as f:
        visited = []
        for filename in files_set:
            vid = filename.split('.npz')[0].split('_')[1]
            if vid not in visited:
                visited.append(vid)  # ignore the video with multiple accidents
            feat_file = os.path.join(features_path, filename)
            data = np.load(feat_file)
            labels = data['labels']  # 2
            toa = 90 if labels[1] > 0 else -1
            tag = 'positive' if labels[1] > 0 else 'negative'
            target_video = os.path.join(data_path, 'videos', '%s/%s/%s.mp4'%(subset, tag, vid))
            assert os.path.exists(target_video), "File does not exist! %s"%(target_video)
            f.writelines('%s/%s/%s %d %d\n'%(subset, tag, vid, labels[1], toa))


def gen_listfiles(data_path, splits_path, val_ratio=0.2):
    
    features_path = os.path.join(data_path, 'features_split', 'training')
    files = os.listdir(features_path)
    all_ids = np.random.permutation(len(files))
    endpoint = int(len(files) * (1-val_ratio))

    # get train set
    files_train = sorted([files[ind] for ind in all_ids[:endpoint]])
    list_file = os.path.join(splits_path, 'train.txt')
    write_listfile(files_train, data_path, features_path, list_file, subset='training')

    # get val set
    files_val = sorted([files[ind] for ind in all_ids[endpoint:]])
    list_file = os.path.join(splits_path, 'val.txt')
    write_listfile(files_val, data_path, features_path, list_file, subset='training')

    # get test set
    features_path = os.path.join(data_path, 'features_split', 'testing')
    files_test = sorted(os.listdir(features_path))
    list_file = os.path.join(splits_path, 'test.txt')
    write_listfile(files_test, data_path, features_path, list_file, subset='testing')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reduce the size of DAD')
    parser.add_argument('--data_path', default="/data/DAD",
                        help='Directory to the original DAD folder.')
    parser.add_argument('--result_path', default="/ssd/data/DAD",
                        help='Directory to the result DAD folder.')
    args = parser.parse_args()

    ratio = 0.5
    n_frames = 100
    FPS = 20
    np.random.seed(123)

    reduce_data(args.data_path, ratio, n_frames, args.result_path)

    # generate splits
    print("generating splits...")
    splits_path = os.path.join(args.result_path, 'splits')
    if not os.path.exists(splits_path):
        os.makedirs(splits_path)
    gen_listfiles(args.data_path, splits_path, val_ratio=0.2)