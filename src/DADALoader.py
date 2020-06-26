import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import src.data_transform

class DADALoader(Dataset):
    def __init__(self, root_path, phase, fps=1, transforms=None, toTensor=False, device=torch.device('cuda')):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.fps = fps
        self.transforms = transforms
        self.toTensor = toTensor
        self.device = device

        self.data_list = self.get_data_list()


    def get_data_list(self):
        # video path
        rgb_path = os.path.join(self.root_path, self.phase, 'rgb')
        assert os.path.exists(rgb_path)
        # loop for each type of accident
        data_list = []
        for accident in sorted(os.listdir(rgb_path)):
            accident_rgb_path = os.path.join(rgb_path, accident)
            for vid in sorted(os.listdir(accident_rgb_path)):
                data_list.append(accident + '/' + vid)
        return data_list

    def read_video_frames(self, video_path, fps=1):
        """ Read video frames
        """
        assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
        frame_ids, video_data = [], []
        for i, filename in enumerate(sorted(os.listdir(video_path))):  # we must sort the image files to ensure a sequential frame order
            if i % fps == 0:
                # get frame id list
                fid = int(filename.split('.')[0])
                frame_ids.append(fid)
                # read video frame
                im = cv2.imread(os.path.join(video_path, filename))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # RGB: (660, 1584, 3)
                video_data.append(im)
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
        return video_data, frame_ids

    def read_focus_images(self, frame_ids, focus_path):
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
            if focus.shape[2] != 1:
                focus = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
            focus_data.append(focus)
        focus_data = np.array(focus_data, dtype=np.float32)
        return focus_data

    def read_coord_arrays(self, frame_ids, coord_file):
        """ Read coordinate array
        """
        assert os.path.exists(coord_file), "File does not exist: %s"%(coord_file)
        coord_data = []
        with open(coord_file, 'r') as f:
            all_lines = f.readlines()
            for fid in frame_ids:
                line = all_lines[fid]
                x_coord = int(line.strip().split(',')[0])
                y_coord = int(line.strip().split(',')[1])
                coord_data.append([x_coord, y_coord])
        coord_data = np.array(coord_data, dtype=np.float32)
        return coord_data


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # video path
        video_path = os.path.join(self.root_path, self.phase, 'rgb', self.data_list[index])
        # focus path
        focus_path = os.path.join(self.root_path, self.phase, 'focus', self.data_list[index])
        # coordinate path
        coord_file = os.path.join(self.root_path, self.phase, 'coordinate', self.data_list[index] + '_coordinate.txt')

        # read video frame data
        video_data, frame_ids = self.read_video_frames(video_path, fps=self.fps)
        if self.transforms is not None:
            video_data = self.transforms(video_data)

        # read focus data
        focus_data = self.read_focus_images(frame_ids, focus_path)
        if self.transforms is not None:
            focus_data = self.transforms(focus_data)
        
        # read coordinates
        coord_data = self.read_coord_arrays(frame_ids, coord_file)

        if self.toTensor:
            video_data = torch.Tensor(video_data).to(self.device)
            focus_data = torch.Tensor(focus_data).to(self.device)
            coord_data = torch.Tensor(coord_data).to(self.device)

        return video_data, focus_data, coord_data
     

class PreFetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_video_data, self.next_focus_data, self.next_coord_data = next(self.loader)
        except StopIteration:
            self.next_video_data = None
            self.next_focus_data = None
            self.next_coord_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_video_data = self.next_video_data.cuda(non_blocking=True)
            self.next_focus_data = self.next_focus_data.cuda(non_blocking=True)
            self.next_coord_data = self.next_coord_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        next_video_data, next_focus_data, next_coord_data = self.next_video_data, self.next_focus_data, self.next_coord_data
        self.preload()
        return next_video_data, next_focus_data, next_coord_data

     
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    p = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transforms = transforms.Compose([data_transform.CenterCrop(224)])

    train_data = DADALoader(p.data_path, 'training', transforms=transforms, toTensor=False, device=device)
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, num_workers=1)
    print("# train set: %d"%(len(train_data)))

    test_data = DADALoader(p.data_path, 'testing', transforms=transforms, toTensor=False, device=device)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=True, num_workers=1)
    print("# test set: %d"%(len(test_data)))

    val_data = DADALoader(p.data_path, 'validation', transforms=transforms, toTensor=False, device=device)
    valdata_loader = DataLoader(dataset=val_data, batch_size=p.batch_size, shuffle=True, num_workers=1)
    print("# val set: %d"%(len(val_data)))

    # prefetcher = PreFetcher(traindata_loader)
    # video_data, focus_data, coord_data = prefetcher.next()
    # iteration = 0
    # while video_data is not None:
    #     iteration += 1
    #     video_data, focus_data, coord_data = prefetcher.next()
    #     print(iteration)

    for i, (video_data, focus_data, coord_data) in enumerate(traindata_loader):
        print("batch: %d / %d"%(i, len(traindata_loader)))
        # video_data = video_data.to(device, non_blocking=True)
        # focus_data = focus_data.to(device, non_blocking=True)
        # coord_data = coord_data.to(device, non_blocking=True)
        # print(video_data.size())  # [batchsize, 342, 660, 1584, 3]
        # print(focus_data.size())
        # print(coord_data.size())

    for i, (video_data, focus_data, coord_data) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):
        pass

    for i, (video_data, focus_data, coord_data) in tqdm(enumerate(valdata_loader), total=len(valdata_loader)):
        pass

