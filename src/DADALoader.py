import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DADALoader(Dataset):
    def __init__(self, root_path, phase, interval=1, max_frames=-1, 
                       transforms={'image':None, 'focus': None, 'fixpt': None}, 
                       params_norm=None, group_classes=False, toTensor=True):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.max_frames = max_frames
        self.transforms = transforms
        self.params_norm = params_norm
        self.group_classes = group_classes
        self.toTensor = toTensor
        self.fps = 30
        self.exclude_accidents = ['52', '53', '54']

        self.data_list = self.get_data_list()
        mapping_file = os.path.join(root_path, 'mapping.txt')
        self.map_dicts = self.read_mapping(mapping_file)


    def get_data_list(self):
        # video path
        rgb_path = os.path.join(self.root_path, self.phase, 'rgb')
        assert os.path.exists(rgb_path), "Path does not eixst! %s"%(rgb_path)
        # loop for each type of accident
        data_list = []
        for accident in sorted(os.listdir(rgb_path)):
            if accident in self.exclude_accidents:
                continue
            accident_rgb_path = os.path.join(rgb_path, accident)
            for vid in sorted(os.listdir(accident_rgb_path)):
                data_list.append(accident + '/' + vid)
        return data_list


    def read_mapping(self, map_file):
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


    def select_frames(self, frame_ids, num=-1):
        if num < 0:
            num = len(frame_ids)
        if num <= len(frame_ids):
            inds = np.random.choice(len(frame_ids), size=num, replace=False)
        else:
            inds = np.random.choice(len(frame_ids), size=num, replace=True)
        inds = np.sort(inds)
        sel_frames = [frame_ids[i] for i in inds]
        return sel_frames, inds


    def read_frames_from_images(self, index, interval=1, max_frames=-1):
        """ Read video frames
        """
        # video path
        video_path = os.path.join(self.root_path, self.phase, 'rgb', self.data_list[index])
        assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
        frame_ids, video_data = [], []
        for i, filename in enumerate(sorted(os.listdir(video_path))):  # we must sort the image files to ensure a sequential frame order
            if i % interval == 0:
                # get frame id list
                fid = int(filename.split('.')[0])
                frame_ids.append(fid)
                # read video frame
                im = cv2.imread(os.path.join(video_path, filename))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # RGB: (660, 1584, 3)
                video_data.append(im)
        if max_frames > 0:
            frame_ids, inds = self.select_frames(frame_ids, num=max_frames)
            video_data = [video_data[i] for i in inds]
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
        return video_data, frame_ids

    def read_frames_from_videos(self, index, interval=1, max_frames=-1):
        """Read video frames
        """
        video_path = os.path.join(self.root_path, self.phase, 'rgb_videos', self.data_list[index] + '.avi')
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
            frame_ids, inds = self.select_frames(frame_ids, num=max_frames)
            video_data = video_data[inds]
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
        return video_data, frame_ids


    def read_focus_from_images(self, frame_ids, index):
        """ Read focus images
        """
        # focus path
        focus_path = os.path.join(self.root_path, self.phase, 'focus', self.data_list[index])
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
            focus = np.expand_dims(focus, axis=-1)  # (H, W, 1)
            focus_data.append(focus)
        focus_data = np.array(focus_data, dtype=np.float32)
        return focus_data


    def read_focus_from_videos(self, frame_ids, index):
        """ Read focus images
        """
        # focus path
        focus_path = os.path.join(self.root_path, self.phase, 'focus_videos', self.data_list[index] + '.avi')
        assert os.path.exists(focus_path), "Path does not exist: %s"%(focus_path)
        focus_data = []
        cap = cv2.VideoCapture(focus_path)
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid-1)
            ret, focus = cap.read()
            assert ret, "read focus video failed! frame: %d"%(fid-1)
            if focus.shape[2] != 1:
                focus = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
            focus = np.expand_dims(focus, axis=-1)  # (H, W, 1)
            focus_data.append(focus)
        focus_data = np.array(focus_data, dtype=np.float32)
        return focus_data


    def read_coord_arrays(self, frame_ids, index):
        """ Read coordinate array
        """
        # coordinate path
        coord_file = os.path.join(self.root_path, self.phase, 'coordinate', self.data_list[index] + '_coordinate.txt')
        assert os.path.exists(coord_file), "File does not exist: %s"%(coord_file)
        # get the class ID
        clsID = self.get_classID(index, group=self.group_classes)
        coord_data = []
        with open(coord_file, 'r') as f:
            all_lines = f.readlines()
            for fid in frame_ids:
                line = all_lines[fid-1]
                x_coord = int(line.strip().split(',')[0])
                y_coord = int(line.strip().split(',')[1])
                cls_label = clsID if x_coord > 0 and y_coord > 0 else 0
                coord_data.append([x_coord, y_coord, cls_label])
        coord_data = np.array(coord_data, dtype=np.float32)
        return coord_data


    def get_classID(self, index, group=False):
        # get accident type in dataset
        atype = self.data_list[index].split('/')[0]
        # map it to the index of accidents in DADA-2000 paper
        idx = self.map_dicts['ID_data'].index(int(atype))
        clsID_paper = self.map_dicts['ID_paper'][idx]
        if group:
            if clsID_paper >= 1 and clsID_paper < 7:
                clsID = 1  # ego dynamic person-centric
            elif clsID_paper >= 7 and clsID_paper < 13:
                clsID = 2  # ego dynamic vehicle-centric
            elif clsID_paper >= 13 and clsID_paper < 19:
                clsID = 3  # ego static road-centric
            elif clsID_paper >= 19 and clsID_paper < 37:
                clsID = 4  # nonego static road-centric
            elif clsID_paper >= 37 and clsID_paper < 52:
                clsID = 5  # nonego dynamic vehicle-centric
            elif clsID_paper >= 52 and clsID_paper < 61:
                clsID = 6  # nonego dynamic person-centric
            else:
                clsID = 3  # accident 51(61)
        else:
            clsID = clsID_paper
        return clsID


    def gather_info(self, index, video_data):
        """Gather info for testing usages
        """
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        nframes = video_data.shape[0]
        height = video_data.shape[1]
        width = video_data.shape[2]
        data_info = np.array([accident_id, video_id, nframes, height, width], dtype=np.int64)
        return data_info


    def __len__(self):
        # return the number of videos for each batch
        return len(self.data_list)

    def __getitem__(self, index):

        # read video frame data, (T, H, W, C)
        video_data, frame_ids = self.read_frames_from_images(index, interval=self.interval, max_frames=self.max_frames)
        # save info
        data_info = self.gather_info(index, video_data)

        # video_data, frame_ids = self.read_frames_from_videos(index, interval=self.interval, max_frames=self.max_frames)
        if self.transforms['image'] is not None:
            video_data = self.transforms['image'](video_data)  # (T, C, H, W)
            if self.params_norm is not None:
                for i in range(video_data.shape[1]):
                    video_data[:, i] = (video_data[:, i] - self.params_norm['mean'][i]) / self.params_norm['std'][i]

        # read focus data, (T, H, W, C)
        focus_data = self.read_focus_from_images(frame_ids, index)
        # focus_data = self.read_focus_from_videos(frame_ids, index)
        if self.transforms['focus'] is not None:
            focus_data = self.transforms['focus'](focus_data)  # (T, 1, H, W)
        
        # read coordinates
        coord_data = self.read_coord_arrays(frame_ids, index)
        if self.transforms['fixpt'] is not None:
            coord_data[:, :2] = self.transforms['fixpt'](coord_data[:, :2])

        if self.toTensor:
            video_data = torch.Tensor(video_data)
            focus_data = torch.Tensor(focus_data)
            coord_data = torch.Tensor(coord_data)
            data_info = torch.Tensor(data_info)

        if self.phase == 'testing':
            return video_data, focus_data, data_info

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


def setup_dataloader(input_shape, output_shape):

    transform_image = transforms.Compose([ProcessImages(input_shape)])
    transform_focus = transforms.Compose([ProcessImages(output_shape)])
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    train_data = DADALoader(args.data_path, 'training', interval=args.frame_interval, max_frames=args.max_frames, 
                            transforms={'image':transform_image, 'focus':transform_focus}, params_norm=params_norm, group_classes=True)
    traindata_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_data = DADALoader(args.data_path, 'validation', interval=args.frame_interval, max_frames=args.max_frames, 
                            transforms={'image':transform_image, 'focus':transform_focus}, params_norm=params_norm, group_classes=True)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    return traindata_loader, evaldata_loader

     
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data_transform import ProcessImages
    import argparse, time
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--max_frames', default=64, type=int,
                        help='Maximum number of frames for each untrimmed video.')
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
    output_shape = (np.array(args.input_shape) / 8).astype(np.int64)
    traindata_loader, evaldata_loader = setup_dataloader(args.input_shape, output_shape)

    # prefetcher = PreFetcher(traindata_loader)
    # video_data, focus_data, coord_data = prefetcher.next()
    # iteration = 0
    # while video_data is not None:
    #     iteration += 1
    #     video_data, focus_data, coord_data = prefetcher.next()
    #     print(iteration)

    num_frames = []
    num_clsses = 6
    cls_stat = np.zeros((num_clsses,), dtype=np.int64)
    t_start = time.time()
    for i, (video_data, focus_data, coord_data) in enumerate(traindata_loader):
        clsID = int(coord_data[0, :, 2].unique()[1])
        print("batch: %d / %d, num_frames = %d, cls = %d, time=%.3f"%(i, len(traindata_loader), video_data.shape[1], clsID, time.time() - t_start))
        cls_stat[clsID-1] += 1
        num_frames.append(video_data.shape[1])
        t_start = time.time()

    np.savez('msg_data', frames=num_frames, stats=cls_stat)

    fig, ax = plt.subplots()
    plt.bar(range(num_clsses), cls_stat)
    plt.xticks(range(num_clsses), ('ego_person', 'ego_vehicle', 'ego_road', 'nonego_road', 'nonego_vehicle', 'nonego_person'))
    plt.title('Number of videos for each category')
    plt.savefig('stat_six.png')