import argparse, math, os
import numpy as np
import yaml
from easydict import EasyDict
from src.enviroment import DashCamEnv
from RLlib.REINFORCE.reinforce_continuous import REINFORCE

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from src.DADALoader import DADALoader
from src.data_transform import ProcessImages, ProcessFixations
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
from metrics.eval_tools import evaluation_accident, evaluation_fixation


def parse_main_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE implementation')
    # For training and testing
    parser.add_argument('--config', default="cfgs/reinforce_default.yml",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
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
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dataloader(cfg, isTraining=True):

    img_shape = [660, 1584]
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape)]),
                      'focus': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # testing dataset
    if not isTraining:
        test_data = DADALoader(cfg.data_path, 'testing', interval=1, max_frames=-1, 
                                transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
        testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader

    # training dataset
    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
    traindata_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # validataion dataset
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames, 
                            transforms=transform_dict, params_norm=params_norm, binary_cls=cfg.binary_cls, use_focus=cfg.use_salmap)
    evaldata_loader = DataLoader(dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print("# train set: %d, eval set: %d"%(len(train_data), len(eval_data)))

    return traindata_loader, evaldata_loader


def train_per_epoch(traindata_loader, env, agent, cfg, writer, epoch):
    # we define each episode as the entire database
    reward_total = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(traindata_loader), 
                                                    total=len(traindata_loader), desc='Training Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        state = env.set_data(video_data, coord_data)
        entropies, log_probs, rewards, states, labels = [], [], [], [], []
        done = False
        rnn_state = Variable(torch.zeros((2, cfg.ENV.batch_size, cfg.REINFORCE.hidden_size))).to(cfg.device) if cfg.REINFORCE.use_lstm else None

        # run each time step
        while not done:
            states.append(state)
            # select action
            action, log_prob, entropy, rnn_state = agent.select_action(state, rnn_state)
            # state transition
            state, reward, done, info = env.step(action)
            
            # gather info
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            labels.append(np.array([env.cur_step-1, env.clsID-1, env.begin_accident, env.fps], dtype=np.float32))

        labels = np.array(labels, dtype=np.float32)
        total_loss, policy_loss, task_loss = agent.update_parameters(rewards, log_probs, entropies, states, rnn_state, labels, cfg.REINFORCE)

        avg_reward = np.sum(rewards) / len(rewards)
        reward_total += np.sum(rewards)

        i_episode = epoch * len(traindata_loader) + i
        writer.add_scalar('loss/total_loss', total_loss, i_episode)
        writer.add_scalar('loss/policy_loss', policy_loss, i_episode)
        writer.add_scalar('loss/task_loss', task_loss, i_episode)
        writer.add_scalar('reward/train_per_video', avg_reward, i_episode)

    writer.add_scalar('reward/train_per_epoch', reward_total, epoch)


def eval_per_epoch(evaldata_loader, env, agent, cfg, writer, epoch):

    total_reward = 0
    for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(evaldata_loader), 
                                                        total=len(evaldata_loader), desc='Testing Epoch: %d / %d'%(epoch + 1, cfg.num_epoch)):  # (B, T, H, W, C)
        # set environment data
        state = env.set_data(video_data, coord_data)
        rewards = []
        done = False
        rnn_state = torch.zeros((2, cfg.ENV.batch_size, cfg.REINFORCE.hidden_size)).to(cfg.device) if cfg.REINFORCE.use_lstm else None

        while not done:
            # select action
            action, log_prob, entropy, rnn_state = agent.select_action(state, rnn_state)
            # state transition
            state, reward, done, info = env.step(action)
            rewards.append(reward)

        avg_reward = np.sum(rewards) / len(rewards)
        total_reward += np.sum(rewards)

        i_episode = epoch * len(evaldata_loader) + i
        writer.add_scalar('reward/test_per_video', avg_reward, i_episode)
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

    # initialize agents
    agent = REINFORCE(env.dim_state, env.dim_action, cfg.REINFORCE, device=cfg.device)
    
    num_episode = 0
    for e in range(cfg.num_epoch):
        # train each epoch
        agent.policy_model.train()
        train_per_epoch(traindata_loader, env, agent, cfg, writer, e)

        # save model file for each epoch (episode)
        torch.save(agent.policy_model.state_dict(), os.path.join(ckpt_dir, 'reinforce_epoch_%02d.pth'%(e+1)))

        # evaluate each epoch
        agent.policy_model.eval()
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

        # initialize agents
        agent = REINFORCE(env.dim_state, env.dim_action, cfg.REINFORCE, device=cfg.device)

        # load agent models (by default: the last epoch)
        ckpt_dir = os.path.join(cfg.output, 'checkpoints')
        agent.load_models(ckpt_dir, cfg)
        
        agent.policy_model.eval()
        save_dict = {'score_preds': [], 'fix_preds': [], 'gt_labels': [], 'gt_fixes': [], 'toas': [], 'vids': []}
        for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):  # (B, T, H, W, C)
            # set environment data
            state = env.set_data(video_data, coord_data)

            done = False
            pred_scores, pred_fixes = [], []
            while not done:
                # select action
                action, _, _ = agent.select_action(state)
                # state transition
                state, reward, done, _ = env.step(action)
                # gather action results
                next_fixation = env.pred_to_point(action[0], action[1])
                score = 0.5 * (action[2] + 1.0)
                pred_fixes.append(next_fixation)
                pred_scores.append(score)
            
            gt_fixes = env.coord_data[:, :2]
            # save results
            save_dict['score_preds'].append(np.array(pred_scores, dtype=np.float32))
            save_dict['fix_preds'].append(np.array(pred_fixes, dtype=np.float32))
            save_dict['gt_labels'].append(env.clsID)
            save_dict['gt_fixes'].append(gt_fixes)
            save_dict['toas'].append(env.begin_accident)
            save_dict['vids'].append(data_info[0, 1])

        np.save(result_file, save_dict)

    # evaluate the results
    gt_labels = save_dict['gt_labels']
    all_fps = [30/cfg.ENV.frame_interval] * len(gt_labels)
    vis_file = os.path.join(output_dir, 'PRCurve_SAC.png')
    AP, mTTA, TTA_R80 = evaluation_accident(save_dict['score_preds'], gt_labels, save_dict['toas'], all_fps, draw_curves=True, vis_file=vis_file)
    # print
    print("AP = %.4f, mean TTA = %.4f, TTA@0.8 = %.4f"%(AP, mTTA, TTA_R80))
    mse_fix = evaluation_fixation(save_dict['fix_preds'], save_dict['gt_fixes'])
    print('Fixation Prediction MSE=%.4f'%(mse_fix))


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_main_args()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        test()
    elif cfg.phase == 'eval':
        # evaluate()
        pass
    else:
        raise NotImplementedError

    