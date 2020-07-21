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
from src.eval_tools import evaluation_dynamic

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


def setup_dataloader(cfg, isTraining=True):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'focus': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # testing dataset
    if not isTraining:
        test_data = DADALoader(cfg.data_path, 'testing', interval=1, max_frames=-1, 
                                transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls)
        testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

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
    writer.add_scalar('loss/task_loss', task_loss, updates)
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
                    if updates % cfg.SAC.logging_interval == 0:
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
        avg_reward = episode_reward / episode_steps
        writer.add_scalar('reward_avg/train', avg_reward, i_episode)

    return updates


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):
    
    for i, (video_data, focus_data, coord_data) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), 
                                                                                    desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, focus_data, coord_data)

        init_state = np.zeros((env.batch_size, cfg.SAC.hidden_size), dtype=np.float32)
        rnn_state = (init_state, init_state)
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            # select action
            action, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
            # step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            # transition
            state = next_state
        
        # logging
        i_episode = epoch * len(evaldata_loader) + i
        avg_reward = episode_reward / episode_steps
        writer.add_scalar('reward_avg/test', avg_reward, i_episode)


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
        agent.save_models(ckpt_dir, cfg, e + 1)

        # evaluate each epoch
        agent.set_status('eval')
        eval_per_epoch(evaldata_loader, env, agent, cfg, writer, e)

    writer.close()
    env.close()
    

def test():
    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, 'results.npy')
    if os.path.exists(result_file):
        save_dict = np.load(result_file, allow_pickle=True)
        save_dict = save_dict.item()
    else:
        # initilize environment
        env = DashCamEnv(cfg.ENV.input_shape, cfg.ENV.dim_action, cfg.ENV.dim_state, fps=30/cfg.ENV.frame_interval, device=cfg.device)
        env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
        cfg.ENV.output_shape = env.output_shape

        # initialize dataset
        testdata_loader = setup_dataloader(cfg.ENV, isTraining=False)

        # AgentENV
        agent = SAC(env.dim_state, env.dim_action, cfg.SAC, device=cfg.device)

        # load agent models (by default: the last epoch)
        ckpt_dir = os.path.join(cfg.output, 'checkpoints')
        agent.load_models(ckpt_dir, cfg)

        # start to test 
        agent.set_status('eval')
        save_dict = {'all_preds': [], 'gt_labels': [], 'toas': [], 'vids': []}
        for i, (video_data, focus_data, coord_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):  # (B, T, H, W, C)
            # set environment data
            state = env.set_data(video_data, focus_data, coord_data)

            # init vars before each episode
            rnn_state = np.zeros((2, env.batch_size, cfg.SAC.hidden_size), dtype=np.float32)
            episode_reward = 0
            done = False
            pred_scores = []
            while not done:
                # select action
                action, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
                next_fixation = env.pred_to_point(action[0], action[1])
                accident_pred = action[2:]
                score = np.exp(accident_pred[1]) / np.sum(np.exp(accident_pred))
                pred_scores.append(score)

                # step
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                # transition
                state = next_state
            # save results
            save_dict['all_preds'].append(np.array(pred_scores, dtype=np.float32))
            save_dict['gt_labels'].append(env.clsID)
            save_dict['toas'].append(env.begin_accident)
            save_dict['vids'].append(data_info[0, 1])
            
        np.save(result_file, save_dict)

    # evaluate the results
    all_pred = save_dict['all_preds']
    all_labels = save_dict['gt_labels']
    all_toas = save_dict['toas']
    all_fps = [30/cfg.ENV.frame_interval] * len(all_labels)
    AP, mTTA, TTA_R80 = evaluation_dynamic(all_pred, all_labels, all_toas, all_fps)
    # print
    print("AP = %.4f, mean TTA = %.4f, TTA@0.8 = %.4f"%(AP, mTTA, TTA_R80))


if __name__ == "__main__":
    
    # parse input arguments
    cfg = parse_configs()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        test()
    else:
        raise NotImplementedError