import argparse, math, os
import numpy as np
from src.enviroment import DashCamEnv
from RLlib.REINFORCE.reinforce_continuous import REINFORCE

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.DADALoader import DADALoader
from src.data_transform import ProcessImages, ProcessFixations


def parse_main_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    # For data loading
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--frame_rate', type=int, default=2,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[240, 320],
                        help='The input shape of images. default: [r=480, c=640]')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--max_frames', default=-1, type=int,
                        help='Maximum number of frames for each untrimmed video. Default: -1 (all frames)')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='How many sub-workers to load dataset. Default: 0')
    # For training and testing
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                        help='number of episodes (default: 2000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--ckpt_freq', type=int, default=100, 
                help='model saving frequency')
    parser.add_argument('--display', type=bool, default=False,
                        help='display or not')
    parser.add_argument('--output', default='./output/REINFORCE',
                        help='Directory of the output. ')
    args = parser.parse_args()
    return args


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dataloader(input_shape, output_shape):

    img_shape = [660, 1584]
    transform_dict = {'image': transforms.Compose([ProcessImages(input_shape)]),
                      'focus': transforms.Compose([ProcessImages(output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(input_shape, img_shape)])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    # training dataset
    train_data = DADALoader(args.data_path, 'training', interval=args.frame_interval, max_frames=args.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, group_classes=True)
    traindata_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # validataion dataset
    eval_data = DADALoader(args.data_path, 'validation', interval=args.frame_interval, max_frames=args.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, group_classes=True)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    return traindata_loader, evaldata_loader


def train():
    # initilize environment
    env = DashCamEnv(args.input_shape, device=device)

    # prepare output directory
    ckpt_dir = os.path.join(args.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # initialize agents
    dim_observe = 128
    dim_action = 8
    agent = REINFORCE(args.hidden_size, dim_observe, dim_action, device=device)

    # initialize dataset
    traindata_loader, evaldata_loader = setup_dataloader(args.input_shape, env.output_shape)
    
    for i_episode in range(args.num_episodes):
        entropies = []
        log_probs = []
        rewards = []
        for i, (video_data, focus_data, coord_data) in enumerate(traindata_loader):  # (B, T, H, W, C)
            print("batch: %d / %d"%(i, len(traindata_loader)))
            env.set_data(video_data, focus_data, coord_data, fps=30/args.frame_interval)

            # run each time step
            for t in range(video_data.size(1)):
                frame_data = video_data[:, t].to(device, dtype=torch.float)  # (B, C, H, W)
                state = env.observe(frame_data)  # (1, 128)
                action, log_prob, entropy = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state.clone()

                if done:
                    break

            agent.update_parameters(rewards, log_probs, entropies, args.gamma)


        if i_episode%args.ckpt_freq == 0:
            torch.save(agent.model.state_dict(), os.path.join(ckpt_dir, 'reinforce-'+str(i_episode)+'.pkl'))

        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
        
    env.close()


if __name__ == "__main__":
    # parse input arguments
    args = parse_main_args()

    # fix random seed 
    set_deterministic(args.seed)

    # gpu options
    gpu_ids = [int(id) for id in args.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.phase == 'train':
        train()
    elif args.phase == 'test':
        # test()
        pass
    elif args.phase == 'eval':
        # evaluate()
        pass
    else:
        raise NotImplementedError

    