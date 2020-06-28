import os 
import torch
from sklearn.utils import shuffle
from src.saliency_models import MLNet, ModMSELoss
# from src.utils import preprocess_images, preprocess_maps
from src.DADALoader import DADALoader
import time, argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data_transform import ProcessImages
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Saliency implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--max_frames', default=16, type=int,
                        help='Maximum number of frames for each untrimmed video.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of training epochs, default: 20.')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[480, 640],
                        help='The input shape of images. default: [r=480, c=640]')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed (default: 123)')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--output', default='./output/saliency',
                        help='Directory of the output. ')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='How many sub-workers to load dataset. Default: 0')
    args = parser.parse_args()

    # fix random seed 
    set_deterministic(args.seed)

    # gpu options
    gpu_ids = [int(id) for id in args.gpus.split(',')]
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # prepare output directory
    ckpt_dir = os.path.join(args.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # tensorboard logging
    tb_dir = os.path.join(args.output, 'tensorboard')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    TBlogger = SummaryWriter(tb_dir)

    # model
    model = MLNet(args.input_shape).to(device)  # ~700MiB
    # freezing Layer
    last_freeze_layer = 23
    # last_freeze_layer = 28
    for i,param in enumerate(model.parameters()):
        if i < last_freeze_layer:
            param.requires_grad = False

    # dataset loader
    transform_image = transforms.Compose([ProcessImages(args.input_shape)])
    transform_focus = transforms.Compose([ProcessImages(model.output_shape)])
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    train_data = DADALoader(args.data_path, 'training', interval=args.frame_interval, max_frames=args.max_frames, shape=args.input_shape, 
                            transforms={'image':transform_image, 'focus':transform_focus}, params_norm=params_norm, toTensor=False)
    traindata_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    eval_data = DADALoader(args.data_path, 'validation', interval=args.frame_interval, max_frames=args.max_frames, shape=args.input_shape, 
                            transforms={'image':transform_image, 'focus':transform_focus}, params_norm=params_norm, toTensor=False)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    # loss (criterion)
    criterion = ModMSELoss(model.output_shape).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

    for k in range(args.epoch):
        t1 = time.time()
        image_trained = 0
        # train the model 
        model.train()
        for i, (video_data, focus_data, _) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc="epoch %d"%(k)):
            optimizer.zero_grad()
            # move data to device (gpu)
            video_data = video_data.view(-1, video_data.size(2), video_data.size(3), video_data.size(4)) \
                        .contiguous().to(device, dtype=torch.float)  # ~30 MiB
            focus_data = focus_data.view(-1, 1, focus_data.size(3), focus_data.size(4)) \
                        .contiguous().to(device, dtype=torch.float)
            # forward
            out = model.forward(video_data)
            # loss
            loss = criterion(out, focus_data, model.prior.clone())
            loss.backward()
            optimizer.step()
            # print
            TBlogger.add_scalars("loss", {'train_loss': loss.item()}, k * args.epoch + i)
            # print("batch: %d / %d, train loss = %.3f"%(i, len(traindata_loader), loss.item()))
            t1 = time.time()

        # eval the model
        model.eval()
        loss_val = 0
        for i, (video_data, focus_data, _) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), desc="epoch %d"%(k)):
            # move data to device (gpu)
            video_data = video_data.view(-1, video_data.size(2), video_data.size(3), video_data.size(4)) \
                        .contiguous().to(device, dtype=torch.float)  # ~30 MiB
            focus_data = focus_data.view(-1, 1, focus_data.size(3), focus_data.size(4)) \
                        .contiguous().to(device, dtype=torch.float)
            with torch.no_grad():
                # forward
                out = model.forward(video_data)
                loss = criterion(out, focus_data, model.prior.clone())
                loss_val += loss.item()
            loss_val /= len(evaldata_loader)
            # write tensorboard logging
            TBlogger.add_scalars("loss", {'eval_loss': loss_val}, k * args.epoch + i)

        # save the model
        model_file = os.path.join(ckpt_dir, 'saliency_model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
    TBlogger.close()
