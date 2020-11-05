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
import random

from src.I3D import InceptionI3d
from src.DADA2KS import DADA2KS
from src.data_transform import ProcessImages
from sklearn.metrics import roc_auc_score


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def setup_dataloader(cfg, isTraining=True):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]), 'salmap': None, 'fixpt': None}
    # testing dataset
    if not isTraining:
        eval_set = 'testing'  # validation
        test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=False, use_fixation=False)
        testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader, test_data.num_classes

    # training dataset
    train_data = DADA2KS(cfg.data_path, 'training', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=False, use_fixation=False)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True)

    # validataion dataset
    eval_data = DADA2KS(cfg.data_path, 'validation', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=False, use_fixation=False)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=cfg.num_workers, pin_memory=True)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader, train_data.num_classes


def setup_i3d(num_classes, weight='i3d_rgb_charades.pt', device=torch.device('cuda')):
    """Setup I3D model"""
    i3d = InceptionI3d(157, in_channels=3, in_temporal=8)  # load the layers before Mixed_5c
    assert os.path.exists(weight), "I3D weight file does not exist! %s"%(weight)
    i3d.load_state_dict(torch.load(weight))
    i3d.replace_logits(num_classes) 
    i3d.to(device)
    i3d = nn.DataParallel(i3d)
    return i3d

def fix_parameters(i3d, train_all=True):
    # fix parameters before Mixed_5 block
    for p in i3d.parameters():
        p.requires_grad = False
    for p_name, p_obj in i3d.named_parameters():
        if not train_all:
            # fine tune training
            if 'Mixed_5' in p_name:
                p_obj.requires_grad = True
            elif 'Logits' in p_name:
                p_obj.requires_grad = True
        else:
            # train all layers
            p_obj.requires_grad = True


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
    traindata_loader, evaldata_loader, num_classes = setup_dataloader(args, isTraining=True)

    # I3D model
    i3d = setup_i3d(num_classes, weight=args.pretrained_i3d, device=device)
    fix_parameters(i3d, train_all=args.train_all)
    i3d.train()

    # optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, i3d.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40])
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, i3d.parameters()), lr=args.learning_rate, weight_decay=1e-5)
    optimizer.zero_grad()

    steps = 0  # total number of gradient steps
    avg_loss = 0  # moving average loss
    for k in range(args.epoch):
        i3d.train()
        for i, (video_data, _, _, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), 
                                                                                     desc='Epoch: %d / %d'%(k + 1, args.epoch)):  # (B, T, H, W, C)
            data_input = Variable(video_data[:, -8:].permute(0, 2, 1, 3, 4)).to(device, non_blocking=True)  # (B, 3, T, H, W)
            label_onehot = torch.zeros(args.batch_size, num_classes)
            label_onehot.scatter_(1, data_info[:, 4].unsqueeze(1).long(), 1)  # one-hot, (B, 2)
            label_onehot = Variable(label_onehot).to(device, non_blocking=True)
            # run forward
            predictions = i3d(data_input)
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(predictions, dim=2)[0], label_onehot)
            loss = cls_loss / args.num_steps
            loss.backward()
            avg_loss += loss.item()

            if (k * len(traindata_loader) + i + 1) % args.num_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                TBlogger.add_scalar('loss/train', avg_loss, steps)
                steps += args.num_steps
                avg_loss = 0  # moving average
        lr_sched.step()
        TBlogger.add_scalar('learning_rate', lr_sched.get_last_lr(), k)

        # save model for this epoch
        if (k+1) % args.snapshot_steps == 0:
            model_file = os.path.join(ckpt_dir, 'i3d_accident_%02d.pth'%(k+1))
            torch.save({'model': i3d.module.state_dict() if len(gpu_ids) > 1 else i3d.state_dict(), 
                        'optimizer': optimizer.state_dict()}, model_file)
                    
        # testing
        i3d.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i, (video_data, _, _, data_info) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), 
                                                                                    desc='Epoch: %d / %d'%(k + 1, args.epoch)):  # (B, T, H, W, C)
                data_input = video_data[:, -8:].permute(0, 2, 1, 3, 4).to(device, non_blocking=True)  # (B, T, 3, H, W)
                label_onehot = torch.zeros(args.batch_size, num_classes)
                label_onehot.scatter_(1, data_info[:, 4].unsqueeze(1).long(), 1)  # one-hot
                label_onehot = label_onehot.to(device, non_blocking=True)
                # run forward
                predictions = i3d(data_input)
                # validation loss
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(predictions, dim=2)[0], label_onehot)
                val_loss += cls_loss.item()
                # # results
                preds = torch.softmax(torch.max(predictions, dim=2)[0], dim=1)
                all_preds.append(preds[:, 1].cpu().numpy())
                all_labels.append(data_info[:, 4].numpy())
        # write logging and evaluate accuracy
        val_loss /= len(evaldata_loader)
        TBlogger.add_scalar('loss/val', val_loss, (k+1) * len(traindata_loader))
        # accuracy
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        AUC_video = roc_auc_score(all_labels, all_preds)
        TBlogger.add_scalar('accuracy/auc', AUC_video, (k+1) * len(traindata_loader))

def test():
    # prepare output directory
    output_dir = os.path.join(args.output, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, 'results.npz')
    if os.path.exists(result_file):
        save_dict = np.load(result_file, allow_pickle=True)
        all_preds, all_labels, all_vids = save_dict['preds'], save_dict['labels'], save_dict['vids']
    else:
        # initialize dataset
        testdata_loader,num_classes = setup_dataloader(args, isTraining=False)

        i3d = setup_i3d(num_classes, weight=args.pretrained_i3d, device=device)
        ckpt = torch.load(args.model_weights)
        i3d.load_state_dict(ckpt['model'])

        i3d.eval()
        all_preds, all_labels, all_vids = [], [], []
        with torch.no_grad():
            for i, (video_data, _, _, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):  # (B, T, H, W, C)
                data_input = video_data[:, -8:].permute(0, 2, 1, 3, 4).to(device, non_blocking=True)  # (B, T, 3, H, W)
                # run forward
                output = i3d(data_input)
                preds = torch.softmax(torch.max(output, dim=2)[0], dim=1)
                preds = preds[:, 1].cpu().numpy()
                labels = data_info[:, 4].numpy()
                vids = data_info[:, :4].numpy()
                all_preds.append(preds)
                all_labels.append(labels)
                all_vids.append(vids)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_vids = np.concatenate(all_vids)
        np.savez(result_file[:-4], preds=all_preds, labels=all_labels, vids=all_vids)
    # evaluation
    AUC_video = roc_auc_score(all_labels, all_preds)
    print("[Correctness] v-AUC = %.5f"%(AUC_video))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch I3D classification implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000-small',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='The number of frames per second for each video. Default: 1')
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--epoch', type=int, default=50,
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
    parser.add_argument('--pretrained_i3d', default='models/i3d_rgb_charades.pt', 
                        help='The model weights for fine-tuning I3D RGB model.')
    parser.add_argument('--model_weights', default=None, 
                        help='The model weights for evaluation or resume training.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The learning rate for fine-tuning.')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='The number of forward steps for each gradient descent update')
    parser.add_argument('--snapshot_steps', type=int, default=5,
                        help='The number of interval steps to save snapshot of trained model.')
    parser.add_argument('--train_all', action='store_true',
                        help='Whether to train all layers or finetune the model.')
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
    else:
        raise NotImplementedError
