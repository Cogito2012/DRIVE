import argparse, math, os
import numpy as np
import random
import yaml
from easydict import EasyDict
from src.enviroment import DashCamEnv
from RLlib.REINFORCE.reinforce import REINFORCE

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from src.DADA2KS import DADA2KS
from src.data_transform import ProcessImages, ProcessFixations
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
from metrics.eval_tools import evaluation_fixation, evaluation_auc_scores, evaluation_accident_new, evaluate_earliness


def parse_main_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    # For training and testing
    parser.add_argument('--config', default="cfgs/reinforce_mlnet.yml",
                        help='Configuration file for REINFORCE algorithm.')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                        help='The ID number of GPU. Default: 0')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
    parser.add_argument('--snapshot_interval', type=int, default=5, metavar='N',
                        help='The epoch interval of model snapshot (default: 5)')
    parser.add_argument('--test_epoch', type=int, default=-1, 
                        help='The snapshot id of trained model for testing.')
    parser.add_argument('--output', default='./output/REINFORCE',
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

def setup_dataloader(cfg, num_workers=0, isTraining=True):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]),
                      'salmap': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    # testing dataset
    if not isTraining:
        test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=cfg.use_salmap)
        testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

    # training dataset
    train_data = DADA2KS(cfg.data_path, 'training', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=cfg.use_salmap, data_aug=cfg.data_aug)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    # validataion dataset
    eval_data = DADA2KS(cfg.data_path, 'validation', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=cfg.use_salmap, data_aug=cfg.data_aug)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader


def train_per_epoch(traindata_loader, env, agent, cfg, writer, epoch):
    # we define each episode as the entire database
    reward_total = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(traindata_loader), 
                                                    total=len(traindata_loader), desc='Training Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        state = env.set_data(video_data, coord_data, data_info)
        entropies, log_probs, rewards, states, all_times, all_fixations = [], [], [], [], [], []
        rnn_state = Variable(torch.zeros((2, cfg.ENV.batch_size, cfg.REINFORCE.hidden_size))).to(cfg.device) if cfg.REINFORCE.use_lstm else None
        # run each time step
        episode_steps = 0
        episode_reward = torch.tensor(0.0).to(cfg.device)
        while episode_steps < env.max_steps:
            states.append(state)
            # select action
            action, log_prob, entropy, rnn_state = agent.select_action(state, rnn_state)
            # state transition
            state, reward, info = env.step(action)
            episode_steps += 1
            episode_reward += reward.sum()
            
            # gather info
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            cur_time = torch.FloatTensor([[(env.cur_step-1) * env.step_size / env.fps]] * cfg.ENV.batch_size)  # (B, 1)
            all_times.append(cur_time)
            # gather GT fixations
            next_step = env.cur_step if episode_steps != env.max_steps else env.cur_step - 1
            gt_fix_next = env.coord_data[:, next_step * env.step_size, :].unsqueeze(1)  # (B, 1, 2)
            all_fixations.append(gt_fix_next)

        all_times = torch.cat(all_times, dim=1).to(cfg.device)
        all_fixations = torch.cat(all_fixations, dim=1)
        # update agent after each episode
        losses = agent.update_parameters(rewards, log_probs, entropies, states, rnn_state, all_times, all_fixations, env, cfg)

        reward_total += episode_reward.cpu().numpy()
        i_episode = epoch * len(traindata_loader) + i
        for (k, v) in losses.items():
            writer.add_scalar('loss/%s'%(k), v, i_episode)
    writer.add_scalar('reward/train_per_epoch', reward_total, epoch)


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):

    total_reward = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(evaldata_loader), 
                                                        total=len(evaldata_loader), desc='Testing Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)
        rnn_state = torch.zeros((2, cfg.ENV.batch_size, cfg.REINFORCE.hidden_size)).to(cfg.device) if cfg.REINFORCE.use_lstm else None
        episode_reward = torch.tensor(0.0).to(cfg.device)
        episode_steps = 0
        while episode_steps < env.max_steps:
            # select action
            action, log_prob, entropy, rnn_state = agent.select_action(state, rnn_state)
            # state transition
            state, reward, info = env.step(action)
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
    traindata_loader, evaldata_loader = setup_dataloader(cfg.ENV, cfg.num_workers)

    # initialize agents
    agent = REINFORCE(cfg.REINFORCE, device=cfg.device)
    
    num_episode = 0
    for e in range(cfg.num_epoch):
        # train each epoch
        agent.set_status('train')
        train_per_epoch(traindata_loader, env, agent, cfg, writer, e)

        if (e+1) % cfg.snapshot_interval == 0:
            # save model file for each epoch (episode)
            torch.save(agent.policy_model.state_dict(), os.path.join(ckpt_dir, 'reinforce_epoch_%02d.pth'%(e+1)))

        # evaluate each epoch
        agent.set_status('eval')
        with torch.no_grad():
            eval_per_epoch(evaldata_loader, env, agent, cfg, writer, e)

    writer.close()
    env.close()


def test_all(testdata_loader, env, agent):
    all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = [], [], [], [], [], []
    for i, (video_data, _, coord_data, data_info) in enumerate(testdata_loader):  # (B, T, H, W, C)
        print("Testing video %d/%d, file: %d/%d.avi, frame #: %d (fps=%.2f)."
            %(i+1, len(testdata_loader), data_info[0, 0], data_info[0, 1], video_data.size(1), 30/cfg.ENV.frame_interval))
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)
        rnn_state = torch.zeros((2, cfg.ENV.batch_size, cfg.REINFORCE.hidden_size)).to(cfg.device) if cfg.REINFORCE.use_lstm else None
        score_pred = np.zeros((cfg.ENV.batch_size, env.max_steps), dtype=np.float32)
        fixation_pred = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
        fixation_gt = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
        i_steps = 0
        while i_steps < env.max_steps:
            # select action
            action, _, _, rnn_state = agent.select_action(state, rnn_state)
            # state transition
            state, reward, info = env.step(action, isTraining=False)
            # gather actions
            score_pred[:, i_steps] = info['pred_score'].cpu().numpy()  # shape=(B,)
            fixation_pred[:, i_steps] = info['pred_fixation'].cpu().numpy()  # shape=(B, 2)
            next_step = env.cur_step if i_steps != env.max_steps - 1 else env.cur_step - 1
            fixation_gt[:, i_steps] = env.coord_data[:, next_step*env.step_size, :].cpu().numpy()
            # next step
            i_steps += 1

        # save results
        all_pred_scores.append(score_pred)  # (B, T)
        all_gt_labels.append(env.clsID.cpu().numpy())  # (B,)
        all_pred_fixations.append(fixation_pred)  # (B, T, 2)
        all_gt_fixations.append(fixation_gt)      # (B, T, 2)
        all_toas.append(env.begin_accident.cpu().numpy())  # (B,)
        all_vids.append(data_info[:,:4].numpy())
    
    all_pred_scores = np.concatenate(all_pred_scores)
    all_gt_labels = np.concatenate(all_gt_labels)
    all_pred_fixations = np.concatenate(all_pred_fixations)
    all_gt_fixations = np.concatenate(all_gt_fixations)
    all_toas = np.concatenate(all_toas)
    all_vids = np.concatenate(all_vids)
    return all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids


def test():
    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, 'results.npz')
    if os.path.exists(result_file):
        save_dict = np.load(result_file, allow_pickle=True)
        all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = \
            save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_fixations'], save_dict['gt_fixations'], save_dict['toas'], save_dict['vids']
    else:
        # initilize environment
        env = DashCamEnv(cfg.ENV, device=cfg.device)
        env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
        cfg.ENV.output_shape = env.output_shape

        # initialize dataset
        testdata_loader = setup_dataloader(cfg.ENV, 0, isTraining=False)

        # initialize agents
        agent = REINFORCE(cfg.REINFORCE, device=cfg.device)

        # load agent models (by default: the last epoch)
        ckpt_dir = os.path.join(cfg.output, 'checkpoints')
        agent.load_models(ckpt_dir, cfg)
        
        agent.set_status('eval')
        with torch.no_grad():
            all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = test_all(testdata_loader, env, agent)
        np.savez(result_file[:-4], pred_scores=all_pred_scores, gt_labels=all_gt_labels, pred_fixations=all_pred_fixations, gt_fixations=all_gt_fixations, toas=all_toas, vids=all_vids)

    # evaluate the results
    FPS = 30/cfg.ENV.frame_interval
    B, T = all_pred_scores.shape
    
    mTTA = evaluate_earliness(all_pred_scores, all_gt_labels, all_toas, fps=FPS, thresh=0.5)
    print("\n[Earliness] mTTA@0.5 = %.4f seconds."%(mTTA))

    AP, p05, r05 = evaluation_accident_new(all_pred_scores, all_gt_labels, all_toas, fps=FPS)
    print("[Correctness] AP = %.4f, precision@0.5 = %.4f, recall@0.5 = %.4f"%(AP, p05, r05))

    AUC_video, AUC_frame = evaluation_auc_scores(all_pred_scores, all_gt_labels, all_toas, FPS, video_len=5, pos_only=True, random=False)
    print("[Correctness] v-AUC = %.5f, f-AUC = %.5f"%(AUC_video, AUC_frame))

    mse_fix = evaluation_fixation(all_pred_fixations, all_gt_fixations)
    print('[Attentiveness] Fixation MSE = %.4f\n'%(mse_fix))


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_main_args()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        test()
    else:
        raise NotImplementedError

    