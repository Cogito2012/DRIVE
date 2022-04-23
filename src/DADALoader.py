import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DADALoader(Dataset):
    def __init__(self, root_path, phase, interval=1, max_frames=-1, 
                       transforms={'image':None, 'salmap': None, 'fixpt': None}, 
                       params_norm=None, binary_cls=False, use_salmap=True, use_fixation=True, cls_task=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.max_frames = max_frames
        self.transforms = transforms
        self.params_norm = params_norm
        self.binary_cls = binary_cls
        self.use_salmap = use_salmap
        self.use_fixation = use_fixation
        self.cls_task = cls_task
        self.fps = 30
        # the specified classes are obtained by stat.py, in which classes with two few samples are filtered out.
        self.accident_classes = ['1', '5', '6', '8', '10', '11', '12', '28', '29', '30', '34', '38', '39', '40', '46', '47', '54']
        self.num_classes = len(self.accident_classes)

        self.data_list = self.get_data_list()
        mapping_file = os.path.join(root_path, 'mapping.txt')
        self.map_dicts = self.read_mapping(mapping_file)


    def get_data_list(self):
        # video path
        rgb_path = os.path.join(self.root_path, self.phase, 'rgb_videos')
        assert os.path.exists(rgb_path), "Path does not eixst! %s"%(rgb_path)
        # loop for each type of accident
        data_list = []
        for accident in sorted(os.listdir(rgb_path)):
            if self.cls_task and accident not in self.accident_classes:
                # for accident classification task, we ignore categories with too few videos
                continue
            accident_rgb_path = os.path.join(rgb_path, accident)
            for vid in sorted(os.listdir(accident_rgb_path)):
                data_list.append(accident + '/' + vid.split('.')[0])
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
            inds = np.arange(len(frame_ids))  # select all T frames
        else:
            inds = np.arange(min(num, len(frame_ids)))  # select top-N or all (N >= T)
        sel_frames = [frame_ids[i] for i in inds]
        return sel_frames, inds


    # def read_frames_from_images(self, index, interval=1, max_frames=-1):
    #     """ Read video frames
    #     """
    #     # video path
    #     video_path = os.path.join(self.root_path, self.phase, 'rgb', self.data_list[index])
    #     assert os.path.exists(video_path), "Path does not exist: %s"%(video_path)
    #     frame_ids, video_data = [], []
    #     for i, filename in enumerate(sorted(os.listdir(video_path))):  # we must sort the image files to ensure a sequential frame order
    #         if i % interval == 0:
    #             # get frame id list
    #             fid = int(filename.split('.')[0])
    #             frame_ids.append(fid)
    #             # read video frame
    #             im = cv2.imread(os.path.join(video_path, filename))
    #             assert im is not None, "Read file failed! %s"%(os.path.join(video_path, filename))
    #             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # RGB: (660, 1584, 3)
    #             video_data.append(im)
    #     if max_frames > 0:
    #         frame_ids, inds = self.select_frames(frame_ids, num=max_frames)
    #         video_data = [video_data[i] for i in inds]
    #     video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
    #     return video_data, frame_ids

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
            video_data = [video_data[i] for i in inds]
        video_data = np.array(video_data, dtype=np.float32)  # 4D tensor, (N, 660, 1584, 3)
        return video_data, frame_ids


    # def read_focus_from_images(self, frame_ids, index):
    #     """ Read focus images
    #     """
    #     # focus path
    #     focus_path = os.path.join(self.root_path, self.phase, 'focus', self.data_list[index])
    #     assert os.path.exists(focus_path), "Path does not exist: %s"%(focus_path)
    #     focus_data = []
    #     for fid in frame_ids:
    #         focus_file = os.path.join(focus_path, "%04d.png"%(fid))
    #         if not os.path.exists(focus_file):
    #             focus_file = os.path.join(focus_path, "%04d.jpg"%(fid))
    #             if not os.path.exists(focus_file):
    #                 print('Focus file does not exist: %s'%(focus_file))
    #                 break
    #         focus = cv2.imread(focus_file)  # BGR: (660, 1584, 3)
    #         if focus.shape[2] != 1:
    #             focus = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
    #         focus = np.expand_dims(focus, axis=-1)  # (H, W, 1)
    #         focus_data.append(focus)
    #     focus_data = np.array(focus_data, dtype=np.float32)
    #     return focus_data


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
        # get the class ID (starting from 1)
        clsID = self.get_classID(index, binary_class=self.binary_cls)
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


    def get_classID(self, index, binary_class=False):
        # get accident type in dataset
        atype = self.data_list[index].split('/')[0]
        # map it to the index of accidents in DADA-2000 paper
        idx = self.map_dicts['ID_data'].index(int(atype))
        clsID_paper = self.map_dicts['ID_paper'][idx]
        if binary_class:
            if clsID_paper < 19 or clsID_paper > 60:
                clsID = 2  # ego-car involved
            else:
                clsID = 1  # ego-car uninvolved
        else:
            clsID = self.accident_classes.index(atype) + 1  # starting from 1
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


    def pre_process(self, video, coord, info, min_len=16):
        # video: (T, 3, H, W)
        # coord: (T, 3)
        # info: (5)
        # return: data_input: (3, 16, H, W), label_input: (1)
        # trim the video
        begin_frame = np.where(coord[:, 2] > 0)[0][0]
        end_frame = np.where(coord[:, 2] > 0)[0][-1]
        len_seg = max(min_len, end_frame - begin_frame + 1)
        # uniform sampling in case of too short video (less than 16 frames)
        inds = np.linspace(begin_frame, end_frame, len_seg).astype(np.int32)
        trimmed_video = np.transpose(video[inds], [1, 0, 2, 3])
        
        # process label
        clsID = self.accident_classes.index(str(int(info[0])))
        logit = np.array([clsID], dtype=np.int32)
        # onehot labels
        onehot_target = np.zeros((self.num_classes))
        onehot_target[clsID] = 1

        return trimmed_video, onehot_target, logit


    def __len__(self):
        # return the number of videos for each batch
        return len(self.data_list)

    def __getitem__(self, index):

        # read video frame data, (T, H, W, C)
        video_data, frame_ids = self.read_frames_from_videos(index, interval=self.interval, max_frames=self.max_frames)
        # video_data, frame_ids = self.read_frames_from_images(index, interval=self.interval, max_frames=self.max_frames)
        # save info
        data_info = self.gather_info(index, video_data)

        if self.transforms['image'] is not None:
            video_data = self.transforms['image'](video_data)  # (T, C, H, W)
            if self.params_norm is not None:
                for i in range(video_data.shape[1]):
                    video_data[:, i] = (video_data[:, i] - self.params_norm['mean'][i]) / self.params_norm['std'][i]

        if self.use_salmap:
            # read focus data, (T, H, W, C)
            # focus_data = self.read_focus_from_images(frame_ids, index)
            focus_data = self.read_focus_from_videos(frame_ids, index)
            if self.transforms['focus'] is not None:
                focus_data = self.transforms['salmap'](focus_data)  # (T, 1, H, W)
        else:
            focus_data = torch.empty(0)
        
        if self.use_fixation:
            # read coordinates
            coord_data = self.read_coord_arrays(frame_ids, index)
            if self.transforms['fixpt'] is not None:
                coord_data[:, :2] = self.transforms['fixpt'](coord_data[:, :2])
        else:
            coord_data = torch.empty(0)

        if self.cls_task:
            data_input, label_target, logit_target = self.pre_process(video_data.astype(np.float32), coord_data, data_info)
            return data_input, label_target, logit_target

        return video_data, focus_data, coord_data, data_info
     

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


def setup_dataloader(cfg):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'focus': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    # training dataset
    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_salmap=cfg.use_salmap)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # validataion dataset
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_salmap=cfg.use_salmap)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    return traindata_loader, evaldata_loader

     
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from .data_transform import ProcessImages, ProcessFixations
    import argparse, time
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--max_frames', default=-1, type=int,
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
    args.image_shape = [330, 792]
    args.output_shape = (np.array(args.input_shape) / 8).astype(np.int64)
    args.binary_cls = True
    args.use_salmap = False
    traindata_loader, evaldata_loader = setup_dataloader(args)

    num_frames = []
    num_clsses = 2
    cls_stat = np.zeros((num_clsses,), dtype=np.int64)
    t_start = time.time()
    for i, (video_data, _, coord_data, data_info) in enumerate(traindata_loader):
        clsID = int(coord_data[0, :, 2].unique()[1])
        print("batch: %d / %d, num_frames = %d, cls = %d, time=%.3f"%(i, len(traindata_loader), video_data.shape[1], clsID, time.time() - t_start))
        cls_stat[clsID-1] += 1
        num_frames.append(video_data.shape[1])
        t_start = time.time()

    fig, ax = plt.subplots()
    plt.bar(range(len(num_frames)), num_frames)
    plt.title('Number of frames for each video')
    plt.savefig('stat_frames_small.png')
    plt.close()

    fig, ax = plt.subplots()
    plt.bar(range(num_clsses), cls_stat)
    # plt.xticks(range(num_clsses), ('ego_person', 'ego_vehicle', 'ego_road', 'nonego_road', 'nonego_vehicle', 'nonego_person'))
    plt.xticks(range(num_clsses), ('ego', 'nonego'))
    plt.title('Number of videos for each category')
    plt.savefig('stat_twocls_small.png')
    plt.close()