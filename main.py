import argparse, os
import torch
import numpy as np
import itertools
import datetime
import random
import yaml
from easydict import EasyDict

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from data_transform import CenterCrop

from DADALoader import DADALoader
from enviroment import FovealVideoEnv
from RLlib.sac import SAC

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/DADA-2000',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--seed', default=12345, type=int,
                        help='The random seed.')

    parser.add_argument('--config', default="cfgs/cfg_sac.yml",
                        help='Configuration file for SAC algorithm.')
    args = parser.parse_args()

    # fix random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    # create custom evironment
    env = FovealVideoEnv(device=device)

    transforms = transforms.Compose([CenterCrop(224)])
    train_data = DADALoader(cfg.data_path, 'training', transforms=transforms, toTensor=False, device=device)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=1)
    print("# train set: %d"%(len(train_data)))

    # Agent
    agent = SAC(env.dim_observation, env.dim_action, cfg)

    #Tesnorboard
    writer = SummaryWriter('logs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cfg.env_name,
                                                                cfg.policy, "autotune" if cfg.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            action = agent.select_action(state)  # Sample action from policy
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state


