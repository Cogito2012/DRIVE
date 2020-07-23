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


def pre_process(data_batch, classes, len_seg=16):
    # video_data: (B, T, 3, H, W)
    # focus_data: empty
    # coord_data: (B, T, 3)
    # data_info: (B, 5)
    # return: data_input: (B, 3, 16, H, W), label_input: (B, 1)
    video_data, _, coord_data, data_info = data_batch
    data_input, label_input, logits = [], [], []
    for i, (video, coord, info) in enumerate(zip(video_data, coord_data, data_info)):
        # trim the video
        begin_frame = torch.nonzero(coord[:, 2] > 0)[0, 0]
        end_frame = torch.nonzero(coord[:, 2] > 0)[-1, 0]
        # uniform sampling (no matter how many frames provided)
        inds = np.linspace(begin_frame, end_frame, len_seg).astype(np.int32)
        trimed_pos = video[inds].permute([1, 0, 2, 3]).unsqueeze(0)  # (1, 3, 16, H, W)
        # sample negative
        if begin_frame - len_seg >= 0:
            # sampling negative at early section
            trimed_neg = video[begin_frame - len_seg: begin_frame].permute([1, 0, 2, 3]).unsqueeze(0)  # (1, 3, 16, H, W)
        elif end_frame + len_seg < video.size(0):
            # sampling negative at later section
            trimed_neg = video[end_frame + 1: end_frame + len_seg + 1].permute([1, 0, 2, 3]).unsqueeze(0)
        data_input.append(torch.cat([trimed_pos, trimed_neg], dim=0))  # (2, 3, 16, H, W))

        # process label
        logit_pos = torch.Tensor([classes.index(str(int(info[0].item()))) + 1]).long()
        logit_neg = torch.zeros_like(logit_pos)
        logit = torch.cat([logit_pos.unsqueeze(0), logit_neg.unsqueeze(0)], dim=0)
        logits.append(logit)

        # onehot labels
        onehot_pos = torch.zeros((len(classes)+1))
        onehot_pos[logit_pos] = 1
        onehot_neg = torch.zeros((len(classes)+1))
        onehot_neg[0] = 1
        label_input.append(torch.cat([onehot_pos.unsqueeze(0), onehot_neg.unsqueeze(0)], dim=0))
    # prepare the input
    data_input = torch.cat(data_input, dim=0)
    label_input = torch.cat(label_input, dim=0)
    return data_input, label_input, logits


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

    steps = 0  # total number of gradient steps
    avg_loss = 0  # moving average loss
    for k in range(args.epoch):
        i3d.train()
        for i, data_batch in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc="Epoch %d [train]"%(k)):
            data_input, label_target, _ = pre_process(data_batch, accident_classes)
            data_input = Variable(data_input.to(device))
            label_target = Variable(label_target.to(device))
            # run forward
            predictions = i3d(data_input)
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(-1), label_target)
            loss = cls_loss / args.num_steps
            loss.backward()
            avg_loss += loss.item()

            if (k * len(traindata_loader) + i + 1) % args.num_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                TBlogger.add_scalar('loss/train', avg_loss, steps)
                steps += 1
                avg_loss = 0  # moving average

        # save model for each epoch
        model_file = os.path.join(ckpt_dir, 'i3d_accident_%02d.pth'%(k+1))
        torch.save({'model': i3d.module.state_dict() if len(gpu_ids) > 1 else i3d.state_dict(), 
                    'optimizer': optimizer.state_dict()}, model_file)
                    
        # testing
        i3d.eval()
        # all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for i, data_batch in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), desc="Epoch %d [eval]"%(k)):
                data_input, label_target, logits = pre_process(data_batch, accident_classes)
                data_input = data_input.to(device)
                # run forward
                predictions = i3d(data_input)
                # logits_pred = torch.argmax(predictions.squeeze(-1), dim=1, keepdim=True)
                # validation loss
                cls_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(-1), label_target)
                val_loss += cls_loss.item()
                # # append results
                # all_preds.append(logits_pred.cpu().numpy())
                # all_labels.append(logits.numpy())
        # write logging and evaluate accuracy
        val_loss /= len(evaldata_loader)
        TBlogger.add_scalar('loss/val', val_loss, (k+1) * len(traindata_loader))
        # all_preds = np.squeeze(np.concatenate(all_preds, axis=0), axis=1)
        # all_labels = np.squeeze(np.concatenate(all_labels, axis=0), axis=1)

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
    parser.add_argument('--input_shape', nargs='+', type=int, default=[224, 224],
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
    parser.add_argument('--num_steps', type=int, default=4,
                        help='The number of forward steps for each gradient descent update')
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
