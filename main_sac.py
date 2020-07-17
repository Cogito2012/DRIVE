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

from src.DADALoader import DADALoader
from src.data_transform import ProcessImages, ProcessFixations
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.enviroment import DashCamEnv
from RLlib.SAC.sac import SAC

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, rnn_state, labels, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, rnn_state, labels, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, rnn_state, labels, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, rnn_state, labels, done

    def __len__(self):
        return len(self.buffer)


def parse_configs():
    parser = argparse.ArgumentParser(description='PyTorch SAC implementation')
    # For training and testing
    parser.add_argument('--config', default="cfgs/sac_default.yml",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
    parser.add_argument('--output', default='./output/SAC',
                        help='Directory of the output. ')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    return cfg


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dataloader(cfg):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'focus': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    # training dataset
    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # validataion dataset
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))
    return traindata_loader, evaldata_loader


def write_logs(writer, outputs, updates):
    """Write the logs to tensorboard"""
    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, task_loss = outputs
    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
    writer.add_scalar('loss/policy', policy_loss, updates)
    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
    writer.add_scalar('entropy_temprature/alpha', task_loss, updates)
    writer.add_scalar('entropy_temprature/alpha', alpha, updates)


def train_per_epoch(traindata_loader, env, agent, cfg, writer, epoch, memory, updates):
    """ Training process for each epoch of dataset
    """
    for i, (video_data, focus_data, coord_data) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), 
                                                                                     desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, focus_data, coord_data)
        episode_reward = 0
        episode_steps = 0
        done = False
        rnn_state = np.zeros((2, env.batch_size, cfg.SAC.hidden_size), dtype=np.float32)

        while not done:
            # select action
            action, rnn_state = agent.select_action(state, rnn_state)

            # Update parameters of all the networks
            if len(memory) > cfg.SAC.batch_size:
                # Number of updates per step in environment
                for _ in range(cfg.SAC.updates_per_step):
                    outputs = agent.update_parameters(memory, cfg.SAC.batch_size, updates)
                    # write log
                    write_logs(writer, outputs, updates)
                    updates += 1

            # step to next state
            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            episode_reward += reward

            # push the current step into memory
            mask = 1 if episode_steps == env.max_step else float(not done)
            labels = np.array([env.cur_step-1, env.clsID, env.begin_accident, env.fps], dtype=np.float32)
            memory.push(state.flatten(), action.flatten(), reward, next_state.flatten(), rnn_state.reshape(-1, cfg.SAC.hidden_size), labels, mask) # Append transition to memory

            # shift to next state
            state = next_state.copy()

        i_episode = epoch * len(traindata_loader) + i
        writer.add_scalar('reward/train', episode_reward, i_episode)
        # print("Episode: {}, episode steps: {}, reward: {}".format(i_episode, episode_steps, round(episode_reward, 2)))
    return updates


def save_models(ckpt_dir, agent, cfg, epoch):
    model_dict = {'policy_model': agent.policy.state_dict(),
                  'policy_optim': agent.policy_optim.state_dict(),
                  'critic_model': agent.critic.state_dict(),
                  'critic_target': agent.critic_target.state_dict(),
                  'configs': cfg}
    if cfg.SAC.automatic_entropy_tuning:
        model_dict.update({'alpha_optim': agent.critic_target.state_dict()})
    torch.save(model_dict, os.path.join(ckpt_dir, 'sac_epoch_%02d.pt'%(epoch)))


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):
    avg_reward = 0.
    for i, (video_data, focus_data, coord_data) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), 
                                                                                    desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, focus_data, coord_data)

        init_state = np.zeros((env.batch_size, cfg.SAC.hidden_size), dtype=np.float32)
        rnn_state = (init_state, init_state)
        episode_reward = 0
        done = False
        while not done:
            # select action
            action, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
            # step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # transition
            state = next_state
        avg_reward += episode_reward
        # logging
        i_episode = epoch * len(evaldata_loader) + i
        writer.add_scalar('episode_reward/test', episode_reward, i_episode)

    avg_reward /= len(evaldata_loader)
    return avg_reward


def train():

    # initilize environment
    env = DashCamEnv(cfg.ENV.input_shape, cfg.ENV.dim_action, cfg.ENV.dim_state, fps=30/cfg.ENV.frame_interval, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    cfg.ENV.output_shape = env.output_shape

    # prepare output directory
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    #Tesnorboard
    writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
    # initialize dataset
    traindata_loader, evaldata_loader = setup_dataloader(cfg.ENV)

    # AgentENV
    agent = SAC(env.dim_state, env.dim_action, cfg.SAC, device=cfg.device)

    # Memory
    memory = ReplayMemory(cfg.SAC.replay_size)

    updates = 0
    for e in range(cfg.num_epoch):
        # train each epoch
        agent.set_status('train')
        updates = train_per_epoch(traindata_loader, env, agent, cfg, writer, e, memory, updates)

        # save model file for each epoch (episode)
        save_models(ckpt_dir, agent, cfg, e + 1)

        # evaluate each epoch
        agent.set_status('eval')
        avg_reward = eval_per_epoch(evaldata_loader, env, agent, cfg, writer, e)

    writer.close()
    env.close()


if __name__ == "__main__":
    
    # parse input arguments
    cfg = parse_configs()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        # test()
        pass
    elif cfg.phase == 'eval':
        # evaluate()
        pass
    else:
        raise NotImplementedError