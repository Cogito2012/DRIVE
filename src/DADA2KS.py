import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DADA2KS(Dataset):
    def __init__(self, root_path, phase, interval=1, transforms={'image':None, 'salmap': None, 'fixpt': None}, 
                       use_salmap=True, use_fixation=True, data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms = transforms
        self.use_salmap = use_salmap
        self.use_fixation = use_fixation
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

    def read_coord_arrays(self, coord_file, start, end):
        """ Read coordinate array
        """
        assert os.path.exists(coord_file), "File does not exist: %s"%(coord_file)
        coord_data = []
        with open(coord_file, 'r') as f:
            all_lines = f.readlines()
            for fid in range(start, end + 1, self.interval):
                line = all_lines[fid-1]
                x_coord = int(line.strip().split(',')[0])
                y_coord = int(line.strip().split(',')[1])
                coord_data.append([x_coord, y_coord])
        coord_data = np.array(coord_data, dtype=np.float32)
        return coord_data

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
        
        # read salmap video (trimmed)
        salmap_data = torch.empty(0)
        if self.use_salmap:
            salmap_path = os.path.join(self.root_path, self.phase, 'salmap_videos', self.data_list[index] + '.avi')
            salmap_data = self.read_video(salmap_path, start, end, toGray=True)
            if self.transforms['salmap'] is not None:
                salmap_data = self.transforms['salmap'](salmap_data)  # (T, 1, H, W)
        
        # read fixation coordinates
        coord_data = torch.empty(0)
        if self.use_fixation:
            coord_file = os.path.join(self.root_path, self.phase, 'coordinate', self.data_list[index] + '_coordinate.txt')
            coord_data = self.read_coord_arrays(coord_file, start, end)
            if self.transforms['fixpt'] is not None:
                coord_data = self.transforms['fixpt'](coord_data)
        
        return video_data, salmap_data, coord_data, data_info



def setup_dataloader(cfg):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'salmap': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    # training dataset
    train_data = DADA2KS(cfg.data_path, 'training', interval=cfg.frame_interval, 
                            transforms=transform_dict, use_salmap=cfg.use_salmap)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    # validataion dataset
    eval_data = DADA2KS(cfg.data_path, 'validation', interval=cfg.frame_interval, 
                            transforms=transform_dict, use_salmap=cfg.use_salmap)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    return traindata_loader, evaldata_loader

     
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from .data_transform import ProcessImages, ProcessFixations
    import argparse, time
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Testing the data loading')
    parser.add_argument('--data_path', default='./data/DADA-2000-small',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[480, 640],
                        help='The input shape of images. default: [r=480, c=640]')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='How many sub-workers to load dataset. Default: 0')
    args = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize dataset
    args.image_shape = [330, 792]
    args.output_shape = (np.array(args.input_shape) / 8).astype(np.int64)
    args.use_salmap = False
    traindata_loader, evaldata_loader = setup_dataloader(args)

    # compute mean & std
    mean_vec = torch.zeros((3), dtype=torch.float32).to(device)
    std_vec = torch.zeros((3), dtype=torch.float32).to(device)
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc='[training set]'):
        # compute mean
        video_data = video_data.to(device, non_blocking=True)  # (B, T, C, H, W)
        mean_vec += torch.mean(video_data, (0, 1, 3, 4))
        std_vec += torch.std(video_data, (0, 1, 3, 4))
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), desc='[validation set]'):
        # compute mean
        video_data = video_data.to(device, non_blocking=True)  # (B, T, C, H, W)
        mean_vec += torch.mean(video_data, (0, 1, 3, 4))
        std_vec += torch.std(video_data, (0, 1, 3, 4))
    mean_vec /= (len(traindata_loader) + len(evaldata_loader))
    std_vec /= (len(traindata_loader) + len(evaldata_loader))
    print('Mean: ', mean_vec.cpu().numpy())   # Mean:  [0.218, 0.220, 0.209]
    print('Std: ', std_vec.cpu().numpy())     # Std:   [0.277, 0.280, 0.277]
    print("Done!")