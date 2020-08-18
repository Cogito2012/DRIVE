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
from metrics.eval_tools import evaluation_accident, evaluation_fixation

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
    parser.add_argument('--test_epoch', type=int, default=-1, 
                        help='The snapshot id of trained model for testing.')
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
        test_data = DADALoader(cfg.data_path, 'testing', interval=cfg.frame_interval, max_frames=-1, 
                                transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
        testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

    # training dataset
    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # validataion dataset
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader


def write_logs(writer, outputs, updates):
    """Write the logs to tensorboard"""
    losses, alpha_values = outputs
    writer.add_scalar('loss/critic', losses['critic'], updates)
    writer.add_scalar('loss/actor', losses['actor'], updates)
    writer.add_scalar('loss/cls', losses['cls'], updates)
    writer.add_scalar('loss/fixation', losses['fixation'], updates)
    writer.add_scalar('loss/policy_total', losses['policy_total'], updates)
    writer.add_scalar('loss/alpha', losses['alpha'], updates)
    writer.add_scalar('temprature/alpha', alpha_values, updates)


def train_per_epoch(traindata_loader, env, agent, cfg, writer, epoch, memory, updates):
    """ Training process for each epoch of dataset
    """
    reward_total = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), 
                                                                                     desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data)
        episode_reward = 0
        episode_steps = 0
        done = False
        rnn_state = np.zeros((2, cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=np.float32)

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
            next_state, reward, done, info = env.step(action) # Step
            episode_steps += 1
            episode_reward += reward

            # push the current step into memory
            mask = 1 if episode_steps == env.max_step else float(not done)
            cur_time = (env.cur_step-1) * env.step_size / env.fps
            gt_fix_next = info['gt_fixation']
            labels = np.array([cur_time, env.clsID-1, env.begin_accident, gt_fix_next[0], gt_fix_next[1]], dtype=np.float32)
            memory.push(state.flatten(), action.flatten(), reward, next_state.flatten(), rnn_state.reshape(-1, cfg.SAC.hidden_size), labels, mask) # Append transition to memory

            # shift to next state
            state = next_state.copy()

        # i_episode = epoch * len(traindata_loader) + i
        # avg_reward = episode_reward / episode_steps
        # writer.add_scalar('reward/train_per_video', avg_reward, i_episode)

        reward_total += episode_reward
    writer.add_scalar('reward/train_per_epoch', reward_total, epoch)

    return updates


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):
    
    total_reward = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), 
                                                                                    desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data)

        rnn_state = np.zeros((2, cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=np.float32)
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            # select action
            action, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
            # step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            # transition
            state = next_state
        
        # logging
        # i_episode = epoch * len(evaldata_loader) + i
        # avg_reward = episode_reward / episode_steps
        # writer.add_scalar('reward/test_per_video', avg_reward, i_episode)

        total_reward += episode_reward
    writer.add_scalar('reward/test_per_epoch', total_reward, epoch)


def train():

    # initilize environment
    env = DashCamEnv(cfg.ENV, device=cfg.device)
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
        with torch.no_grad():
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
        env = DashCamEnv(cfg.ENV, device=cfg.device)
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
        save_dict = {'score_preds': [], 'fix_preds': [], 'gt_labels': [], 'gt_fixes': [], 'toas': [], 'vids': []}
        for i, (video_data, _, coord_data, data_info) in enumerate(testdata_loader):  # (B, T, H, W, C)
            print("Testing video %d/%d, file: %d/%d.avi, frame #: %d (fps=%.2f)."
                %(i+1, len(testdata_loader), data_info[0, 0], data_info[0, 1], video_data.size(1), 30/cfg.ENV.frame_interval))
            # set environment data
            state = env.set_data(video_data, coord_data)

            # init vars before each episode
            rnn_state = np.zeros((2, cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=np.float32)
            done = False
            pred_scores, pred_fixes, gt_fixes = [], [], []
            while not done:
                # select action
                action, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
                next_fixation = env.scales_to_point(action[:2])
                pred_fixes.append(next_fixation)
                # gather ground truth of next fixation
                gt_fixes.append(env.coord_data[(env.cur_step + 1) * env.step_size, :2])

                # accident_pred = action[2:]
                # score = np.exp(accident_pred[1]) / np.sum(np.exp(accident_pred))
                # pred_scores.append(score)
                score = 0.5 * (action[2] + 1.0)
                pred_scores.append(score)

                # step
                next_state, reward, done, _ = env.step(action)
                # transition
                state = next_state
            # gt_fixes = env.coord_data[:, :2]
            gt_fixes = np.vstack(gt_fixes)
            # save results
            save_dict['score_preds'].append(np.array(pred_scores, dtype=np.float32))
            save_dict['fix_preds'].append(np.array(pred_fixes, dtype=np.float32))
            save_dict['gt_labels'].append(env.clsID)
            save_dict['gt_fixes'].append(gt_fixes)
            save_dict['toas'].append(env.begin_accident)
            save_dict['vids'].append(data_info[0, 1])
            
        np.save(result_file, save_dict)

    # evaluate the results
    score_preds = save_dict['score_preds']
    gt_labels = save_dict['gt_labels']
    toas = save_dict['toas']
    all_fps = [30/cfg.ENV.frame_interval] * len(gt_labels)
    vis_file = os.path.join(output_dir, 'PRCurve_SAC.png')
    AP, mTTA, TTA_R80 = evaluation_accident(score_preds, gt_labels, toas, all_fps, draw_curves=True, vis_file=vis_file)
    # print
    print("AP = %.4f, mean TTA = %.4f, TTA@0.8 = %.4f"%(AP, mTTA, TTA_R80))
    mse_fix = evaluation_fixation(save_dict['fix_preds'], save_dict['gt_fixes'])
    print('Fixation Prediction MSE=%.4f'%(mse_fix))


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