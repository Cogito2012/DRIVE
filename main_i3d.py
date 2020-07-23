###########################
# This code script aims to train an I3D model for extracting traffic features of video clip
###########################
import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.I3D import InceptionI3d
from src.DADALoader import DADALoader
from src.data_transform import ProcessImages, ProcessFixations


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dataloader(cfg, isTraining=True):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, [660, 1584])])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # testing dataset
    if not isTraining:
        test_data = DADALoader(cfg.data_path, 'testing', interval=1, max_frames=-1, 
                                transforms=transform_dict, params_norm=params_norm, use_focus=False)
        testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

    # training dataset
    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, use_focus=False)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    accident_classes = train_data.accident_classes

    # validataion dataset
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, use_focus=False)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader, accident_classes


def pre_process(data_batch, classes):
    # video_data: (B, T, 3, H, W)
    # focus_data: empty
    # coord_data: (B, T, 3)
    # data_info: (B, 5)
    # return: data_input: (B, 3, T, H, W), label_input: (B, 1)
    video_data, _, coord_data, data_info = data_batch
    data_input, label_input = [], []
    for i, (video, coord, info) in enumerate(zip(video_data, coord_data, data_info)):
        # trim the video
        begin_frame = torch.nonzero(coord[:, 2] > 0)[0, 0]
        end_frame = torch.nonzero(coord[:, 2] > 0)[-1, 0]
        trimed_video = video[begin_frame: end_frame, :]  # (T, 3, H, W)
        data_input.append(trimed_video.permute([1, 0, 2, 3]).unsqueeze(0))
        # process label
        label = torch.Tensor([classes.index(str(int(info[0].item()))) + 1])
        label_input.append(label.unsqueeze(0))
    # prepare the input
    data_input = torch.cat(data_input, dim=0)
    label_input = torch.cat(label_input, dim=0)
    return Variable(data_input.to(device)), Variable(label_input.to(device))


def train():
    # prepare output directory
    ckpt_dir = os.path.join(args.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # tensorboard logging
    tb_dir = os.path.join(args.output, 'tensorboard')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    TBlogger = SummaryWriter(tb_dir)
    
    # setup dataset loader
    traindata_loader, evaldata_loader, accident_classes = setup_dataloader(args, isTraining=True)
    num_classes = len(accident_classes) + 1  # we add the background class (ID=0)

    # I3D model
    i3d = InceptionI3d(157, in_channels=3)  # load the layers before Mixed_5c
    assert os.path.exists(args.pretrained_i3d), "I3D weight file does not exist! %s"%(args.pretrained_i3d)
    i3d.load_state_dict(torch.load(args.pretrained_i3d))
    i3d.replace_logits(num_classes) 
    i3d.to(device)
    i3d = nn.DataParallel(i3d)
    i3d.train()
    # fix parameters before Mixed_5 block
    for p_name, p_obj in i3d.named_parameters():
        if 'Mixed_5' not in p_name:
            p_obj.requires_grad = False
    # optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, i3d.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-6)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    for k in range(args.epoch):
        i3d.train()
        for i, data_batch in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc="Epoch %d [train]"%(k)):
            data_input, data_label = pre_process(data_batch, accident_classes)
            
        print("Done")

def test():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch I3D classification implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='The number of frames per second for each video. Default: 1')
    parser.add_argument('--max_frames', default=-1, type=int,
                        help='Maximum number of frames for each untrimmed video.')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'eval'],
                        help='Training or testing phase.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of training epochs, default: 20.')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[480, 640],
                        help='The input shape of images. default: [r=480, c=640]')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed (default: 123)')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--output', default='./output/I3D',
                        help='Directory of the output. ')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='How many sub-workers to load dataset. Default: 0')
    parser.add_argument('--pretrained_i3d', default='./i3d_rgb_charades.pt', 
                        help='The model weights for fine-tuning I3D RGB model.')
    parser.add_argument('--model_weights', default=None, 
                        help='The model weights for evaluation or resume training.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The learning rate for fine-tuning.')
    args = parser.parse_args()

    # fix random seed 
    set_deterministic(args.seed)

    # gpu options
    gpu_ids = [int(id) for id in args.gpus.split(',')]
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.phase == 'train':
        train()
    elif args.phase == 'test':
        test()
    elif args.phase == 'eval':
        evaluate()
    else:
        raise NotImplementedError
