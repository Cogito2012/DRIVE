import os
import numpy as np
import cv2
import torch
import random
from torch.utils.data import Dataset


class DADA2KS(Dataset):
    def __init__(self, root_path, phase, interval=1, transforms={'image':None}, data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.num_classes = 2
        # read samples list
        self.data_list, self.labels, self.clips, self.toas = self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(self.root_path, self.phase, self.phase + '.txt')
        assert os.path.exists(list_file), "File does not exist! %s"%(list_file)
        fileIDs, labels, clips, toas = [], [], [], []
        samples_visited, visit_rows = [], []
        with open(list_file, 'r') as f:
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split(' ')  # e.g.: 1/002 1 0 149 136
                fileIDs.append(sample[0])       # 1/002
                labels.append(int(sample[1]))   # 1: positive, 0: negative
                clips.append([int(sample[2]), int(sample[3])])  # [start frame, end frame]
                toas.append(int(sample[4]))     # time-of-accident (toa)
                sample_id = sample[0] + '_' + sample[1]
                if sample_id not in samples_visited:
                    samples_visited.append(sample_id)
                    visit_rows.append(ids)
        if not self.data_aug:
            fileIDs = [fileIDs[i] for i in visit_rows]
            labels = [labels[i] for i in visit_rows]
            clips = [clips[i] for i in visit_rows]
            toas = [toas[i] for i in visit_rows]
        return fileIDs, labels, clips, toas

    def __len__(self):
        return len(self.data_list)

    def read_video(self, video_file, start, end, toGray=False):
        """Read video frames
        """
        assert os.path.exists(video_file), "Path does not exist: %s"%(video_file)
        # get the video data
        video_data = []
        cap = cv2.VideoCapture(video_file)
        for fid in range(start, end + 1, self.interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            assert ret, "read video failed! file: %s frame: %d"%(video_file, fid)
            if toGray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame, axis=-1)  # (H, W, 1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(frame)
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor
        return video_data

    def gather_info(self, index):
        """Gather info for testing usages
        """
        # video file info
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        # toa info
        start, end = self.clips[index]
        if self.labels[index] > 0:  # positive sample
            assert self.toas[index] >= start and self.toas[index] <= end, "sample id: %s"%(self.data_list[index])
            toa = int((self.toas[index] - start) / self.interval)
        else:
            toa = int(self.toas[index])  # negative sample (toa=-1)
        data_info = np.array([accident_id, video_id, start, end, self.labels[index], toa], dtype=np.int32)
        return data_info

    def __getitem__(self, index):
        # clip start and ending
        start, end = self.clips[index]

        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path, self.phase, 'rgb_videos', self.data_list[index] + '.avi')
        video_data = self.read_video(video_path, start, end)
        # gather info
        data_info = self.gather_info(index)
        # pre-process
        if self.transforms['image'] is not None:
            video_data = self.transforms['image'](video_data)  # (T, C, H, W)
        
        return video_data, data_info


class DADDataset(Dataset):
    def __init__(self, root_path, phase, interval=1, transforms=None):
        self.root_path = root_path
        self.phase = phase  # 'train', 'test', 'val'
        self.subset = 'testing' if phase == 'test' else 'training'
        self.interval = interval
        self.transforms = transforms
        self.fps = 20
        self.num_frames = 100
        self.num_classes = 2
        # read samples list
        self.data_list, self.labels, self.toas = self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(self.root_path, 'splits', self.phase + '.txt')
        assert os.path.exists(list_file), "File does not exist! %s"%(list_file)
        fileIDs, labels, toas = [], [], []
        with open(list_file, 'r') as f:
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split(' ')
                # parse file IDs (training/negative/001355)
                fileIDs.append(sample[0])
                # parse label (1: positive, 0: negative)
                label = int(sample[1])
                labels.append(label)
                # get time-of-accident (toa or -1)
                toas.append(int(sample[2]))
        return fileIDs, labels, toas

    def __len__(self):
        return len(self.data_list)

    def read_video(self, video_file):
        assert os.path.exists(video_file), "Path does not exist: %s"%(video_file)
        # get the video data
        video_data = []
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()
        counter = 0
        while (ret):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(frame)
            ret, frame = cap.read()
            counter += 1
        assert counter == self.num_frames, "invalid video file! %s"%(video_file)
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor
        return video_data

    def gather_info(self, index):
        """Gather info for testing usages
        """
        # video file info
        accident_id = 1 if self.labels[index] > 0 else 0
        video_id = int(self.data_list[index].split('/')[2])  # training/positive/001324
        # toa info
        if self.labels[index] > 0:  # positive sample
            toa = int(self.toas[index] / self.interval)
        else:
            toa = int(self.toas[index])  # negative sample (toa=-1)
        data_info = np.array([accident_id, video_id, 0, self.num_frames-1, self.labels[index], toa], dtype=np.int32)
        return data_info

    def __getitem__(self, index):
        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path, self.data_list[index] + '.mp4')
        video_data = self.read_video(video_path)
        # gather info
        data_info = self.gather_info(index)
        # pre-process
        if self.transforms is not None:
            video_data = self.transforms(video_data)  # (T, C, H, W)

        return video_data, data_info



def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from data_setup import setup_dada2ks, setup_ccd, setup_dad
    import argparse, time
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Testing the data loading')
    parser.add_argument('--dataset', default='DADA2KS', choices=['DADA2KS', 'CCD', 'DAD'])
    parser.add_argument('--data_path', default='./data/DADA-2000-small',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'val'],
                        help='Training or testing phase.')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[480, 640],
                        help='The input shape of images. default: [r=480, c=640]')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='How many sub-workers to load dataset. Default: 0')
    args = parser.parse_args()

    set_deterministic(123)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize dataset
    args.output_shape = (np.array(args.input_shape) / 8).astype(np.int64)
    if args.dataset == 'DADA2KS':
        traindata_loader, evaldata_loader = setup_dada2ks(DADA2KS, args)
    elif args.dataset == 'DAD':
        traindata_loader, evaldata_loader = setup_dad(DADDataset, args)

    # compute mean & std
    mean_vec = torch.zeros((3), dtype=torch.float32).to(device)
    std_vec = torch.zeros((3), dtype=torch.float32).to(device)
    for i, (video_data, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc='[training set]'):
        # compute mean
        video_data = video_data.to(device, non_blocking=True)  # (B, T, C, H, W)
        mean_vec += torch.mean(video_data, (0, 1, 3, 4))
        std_vec += torch.std(video_data, (0, 1, 3, 4))
    for i, (video_data, data_info) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), desc='[validation set]'):
        # compute mean
        video_data = video_data.to(device, non_blocking=True)  # (B, T, C, H, W)
        mean_vec += torch.mean(video_data, (0, 1, 3, 4))
        std_vec += torch.std(video_data, (0, 1, 3, 4))
    mean_vec /= (len(traindata_loader) + len(evaldata_loader))
    std_vec /= (len(traindata_loader) + len(evaldata_loader))
    print('Mean: ', mean_vec.cpu().numpy())   # Mean:  [0.218, 0.220, 0.209] (DADA2KS), [0.236, 0.244, 0.246] (CCD)
    print('Std: ', std_vec.cpu().numpy())     # Std:   [0.277, 0.280, 0.277] (DADA2KS), [0.261, 0.273, 0.284] (CCD)
    print("Done!")