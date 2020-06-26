import argparse, math, os
import numpy as np
from src.enviroment import FovealVideoEnv
from RLlib.REINFORCE.reinforce_continuous import REINFORCE

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_transform import CenterCrop

from src.DADALoader import DADALoader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--frame_rate', type=int, default=2,
                        help='The number of frames per second for each video. Default: 10')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
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

    # fix random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initilize environment
    env = FovealVideoEnv(device=device)
    # env.seed(args.seed)

    # prepare output directory
    ckpt_dir = os.path.join(args.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # initialize agents
    agent = REINFORCE(args.hidden_size, 256, 8)
    # initialize dataset
    transforms = transforms.Compose([CenterCrop(224)])
    train_data = DADALoader(args.data_path, 'training', fps=args.frame_rate, transforms=transforms, toTensor=False, device=device)
    traindata_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print("# train set: %d"%(len(train_data)))

    for i_episode in range(args.num_episodes):
        
        entropies = []
        log_probs = []
        rewards = []

        for i, (video_data, focus_data, coord_data) in enumerate(traindata_loader):  # (B, T, H, W, C)
            print("batch: %d / %d"%(i, len(traindata_loader)))
            env.set_data(video_data)

            # run each time step
            for t in range(video_data.size(1)):
                state = env.observe(video_data[:, t])
                action, log_prob, entropy = agent.select_action(state)
                action = action.cpu()

                next_state, reward, done, _ = env.step(action.numpy()[0])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = torch.Tensor([next_state])

                if done:
                    break

            agent.update_parameters(rewards, log_probs, entropies, args.gamma)


        if i_episode%args.ckpt_freq == 0:
            torch.save(agent.model.state_dict(), os.path.join(ckpt_dir, 'reinforce-'+str(i_episode)+'.pkl'))

        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
        
    env.close()