import argparse, os
import torch
import numpy as np
import itertools
import datetime
import random
import yaml
from easydict import EasyDict
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
from torchvision import transforms

from src.DADA2KS import DADA2KS
from src.data_transform import ProcessImages, ProcessFixations
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.enviroment import DashCamEnv
from RLlib.SAC.sac import SAC
from RLlib.SAC.replay_buffer import ReplayMemory, ReplayMemoryGPU
from metrics.eval_tools import evaluation_accident, evaluation_fixation


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
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    return cfg


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
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]),
                      'focus': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    # testing dataset
    if not isTraining:
        test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict, use_focus=cfg.use_salmap)
        testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

    # training dataset
    train_data = DADA2KS(cfg.data_path, 'training', interval=cfg.frame_interval, transforms=transform_dict, use_focus=cfg.use_salmap)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True)

    # validataion dataset
    eval_data = DADA2KS(cfg.data_path, 'validation', interval=cfg.frame_interval, transforms=transform_dict, use_focus=cfg.use_salmap)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=cfg.num_workers, pin_memory=True)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader


def write_logs(writer, outputs, updates):
    """Write the logs to tensorboard"""
    losses, alpha_values = outputs
    for (k, v) in losses.items():
        writer.add_scalar('loss/%s'%(k), v, updates)
    writer.add_scalar('temprature/alpha', alpha_values, updates)


def train_per_epoch(traindata_loader, env, agent, cfg, writer, epoch, memory, updates):
    """ Training process for each epoch of dataset
    """
    reward_total = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), 
                                                                                     desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)

        episode_reward = torch.tensor(0.0).to(cfg.device)
        done = torch.ones((cfg.ENV.batch_size, 1), dtype=torch.float32).to(cfg.device)
        rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                     torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
        episode_steps = 0
        while episode_steps < env.max_steps:
            # select action
            actions, rnn_state = agent.select_action(state, rnn_state)

            # Update parameters of all the networks
            if len(memory) > cfg.SAC.batch_size:
                for _ in range(cfg.SAC.updates_per_step):
                    outputs = agent.update_parameters(memory, updates)
                    if updates % cfg.SAC.logging_interval == 0:
                        # write log
                        write_logs(writer, outputs, updates)
                    updates += 1

            # step to next state
            next_state, rewards = env.step(actions) # Step
            episode_steps += 1
            episode_reward += rewards.sum()

            # push the current step into memory
            mask = done if episode_steps == env.max_steps else done - 1.0
            memory.push(state, actions, rewards, next_state, rnn_state, mask) # Append transition to memory
            # shift to next state
            state = next_state.clone()

        reward_total += episode_reward.cpu().numpy()
    writer.add_scalar('reward/train_per_epoch', reward_total, epoch)
    
    return updates


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):
    
    total_reward = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), 
                                                                                    desc='Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)
        rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                     torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
        episode_reward = torch.tensor(0.0).to(cfg.device)
        episode_steps = 0
        while episode_steps < env.max_steps:
            # select action
            actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
            # step
            state, reward = env.step(actions)
            episode_reward += reward.sum()
            episode_steps += 1

        total_reward += episode_reward.cpu().numpy()
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
    # backup the config file
    with open(os.path.join(cfg.output, 'cfg.yml'), 'w') as bkfile:
        yaml.dump(cfg, bkfile, default_flow_style=False)
    
    # initialize dataset
    traindata_loader, evaldata_loader = setup_dataloader(cfg.ENV)

    # AgentENV
    agent = SAC(env.mask_size, env.output_shape, cfg.SAC, device=cfg.device)

    # Memory
    memory = ReplayMemory(cfg.SAC.replay_size) if not cfg.SAC.gpu_replay else ReplayMemoryGPU(cfg.SAC, cfg.ENV.batch_size, device=cfg.device)

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
    

def test_all(testdata_loader, env, agent):
    all_pred_scores, all_gt_labels, all_pred_masks, all_gt_masks, all_toas, all_vids = [], [], [], [], [], []
    for i, (video_data, _, coord_data, data_info) in enumerate(testdata_loader):  # (B, T, H, W, C)
        print("Testing video %d/%d, file: %d/%d.avi, frame #: %d (fps=%.2f)."
            %(i+1, len(testdata_loader), data_info[0, 0], data_info[0, 1], video_data.size(1), 30/cfg.ENV.frame_interval))
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)

        # init vars before each episode
        pred_scores, pred_masks, gt_masks = [], [], []
        rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                        torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
        score_pred = np.zeros((cfg.ENV.batch_size, env.max_steps), dtype=np.float32)
        mask_pred = np.zeros((cfg.ENV.batch_size, env.max_steps, env.mask_size[0], env.mask_size[1]), dtype=np.float32)
        mask_gt = np.zeros_like(mask_pred, dtype=np.float32)
        i_steps = 0
        while i_steps < env.max_steps:
            # select action
            actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)

            # gather actions
            score_pred[:, i_steps] = 0.5 * (actions[:, 0].cpu().numpy() + 1.0)  # map to [0, 1], shape=(B,)
            mask_pred[:, i_steps] = actions[:, 1:].view(-1, env.mask_size[0], env.mask_size[1]).cpu().numpy()  # shape=(B, 1, 5, 12)
            mask_gt[:, i_steps] = env.mask_data[:, (env.cur_step + 1)*env.step_size, :, :].cpu().numpy()

            # step
            state, reward = env.step(actions, isTraining=False)
            i_steps += 1

        # save results
        all_pred_scores.append(score_pred)  # (B, T)
        all_gt_labels.append(env.clsID.cpu().numpy())  # (B,)
        all_pred_masks.append(mask_pred)  # (B, T, 5, 12)
        all_gt_masks.append(mask_gt)      # (B, T, 5, 12)
        all_toas.append(env.begin_accident.cpu().numpy())  # (B,)
        all_vids.append(data_info[:,:4].numpy())
    
    all_pred_scores = np.concatenate(all_pred_scores)
    all_gt_labels = np.concatenate(all_gt_labels)
    all_pred_masks = np.concatenate(all_pred_masks)
    all_gt_masks = np.concatenate(all_gt_masks)
    all_toas = np.concatenate(all_toas)
    all_vids = np.concatenate(all_vids)
    return all_pred_scores, all_gt_labels, all_pred_masks, all_gt_masks, all_toas, all_vids

def test():
    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, 'results.npz')
    if os.path.exists(result_file):
        save_dict = np.load(result_file, allow_pickle=True)
        all_pred_scores, all_gt_labels, all_pred_masks, all_gt_masks, all_toas, all_vids = \
            save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_masks'], save_dict['gt_masks'], save_dict['toas'], save_dict['vids']
    else:
        # initilize environment
        env = DashCamEnv(cfg.ENV, device=cfg.device)
        env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
        cfg.ENV.output_shape = env.output_shape

        # initialize dataset
        testdata_loader = setup_dataloader(cfg.ENV, isTraining=False)

        # AgentENV
        agent = SAC(env.mask_size, env.output_shape, cfg.SAC, device=cfg.device)

        # load agent models (by default: the last epoch)
        ckpt_dir = os.path.join(cfg.output, 'checkpoints')
        agent.load_models(ckpt_dir, cfg)

        # start to test 
        agent.set_status('eval')
        with torch.no_grad():
            all_pred_scores, all_gt_labels, all_pred_masks, all_gt_masks, all_toas, all_vids = test_all(testdata_loader, env, agent)
        np.savez(result_file[:-4], pred_scores=all_pred_scores, gt_labels=all_gt_labels, pred_masks=all_pred_masks, gt_masks=all_gt_masks, toas=all_toas, vids=all_vids)

    # evaluate the results
    AP, mTTA, TTA_R80, p05, r05, t05 = evaluation_accident(all_pred_scores, all_gt_labels, all_toas, fps=30/cfg.ENV.frame_interval)
    print("AP = %.4f, mean TTA = %.4f, TTA@0.8 = %.4f"%(AP, mTTA, TTA_R80))
    print("\nprecision@0.5 = %.4f, recall@0.5 = %.4f, TTA@0.5 = %.4f\n"%(p05, r05, t05))
    


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