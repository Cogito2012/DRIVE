import os
import torch
import numpy as np
from tqdm import tqdm
from src.enviroment import DashCamEnv
from RLlib.SAC.sac import SAC
from main_sac import parse_configs, set_deterministic
from torch.utils.data import DataLoader
from torchvision import transforms
from src.DADA2KS import DADA2KS
from src.data_transform import ProcessImages, ProcessFixations, padding_inv
import metrics.saliency.saliency_metrics as metrics


def setup_dataloader(cfg, num_workers=0, isTraining=False):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]), 'salmap': None,  
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict)
    testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    print("# test set: %d"%(len(test_data)))
    return testdata_loader


def eval_video_saliency(pred_salmaps, gt_salmaps, toa_batch=None):
    num_videos, num_frames = gt_salmaps.shape[:2]
    metrics_video = np.zeros((num_videos, num_frames, 4), dtype=np.float32)
    for i in range(num_videos):
        for j in range(num_frames):
            map_pred = pred_salmaps[i, j]
            map_gt = gt_salmaps[i, j]
            # We cannot compute AUC metrics (AUC-Judd, shuffled AUC, and AUC_borji) 
            # since we do not have binary map of human fixation points
            sim = metrics.similarity(map_pred, map_gt)
            cc = metrics.cc(map_pred, map_gt)
            nss = metrics.nss(map_pred, map_gt)
            kl = metrics.kldiv(map_pred, map_gt)
            metrics_video[i, j, :] = np.array([sim, cc, nss, kl], dtype=np.float32)
    return metrics_video

def test_saliency():

    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval-saliency')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize environment
    cfg.ENV.use_salmap = True
    env = DashCamEnv(cfg.ENV, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    # cfg.ENV.output_shape = env.output_shape
    height, width = cfg.ENV.image_shape

    # initialize dataset
    testdata_loader = setup_dataloader(cfg.ENV, 0, isTraining=False)
    # AgentENV
    agent = SAC(cfg.SAC, device=cfg.device)
    # load agent models (by default: the last epoch)
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    agent.load_models(ckpt_dir, cfg)
    agent.set_status('eval')

    all_results = []
    # start to test 
    with torch.no_grad():
        for i, (video_data, salmap_data, coord_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):  # (B, T, H, W, C)
            # set environment data
            state = env.set_data(video_data, coord_data, data_info)
            # init vars before each episode
            rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                            torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
            salmaps_pred = []
            i_steps = 0
            while i_steps < env.max_steps:
                salmaps = env.cur_saliency.squeeze(1).cpu().numpy()  # (B, 60, 80)
                salmaps = np.array([padding_inv(sal, height, width) for sal in salmaps], dtype=np.uint8)  # (B, 330, 792)
                salmaps_pred.append(np.expand_dims(salmaps, axis=1))
                # select action
                actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
                # step
                state, reward, info = env.step(actions, isTraining=False)
                i_steps += 1
            salmaps_pred = np.concatenate(salmaps_pred, axis=1)  # (B, T, 330, 792)
            salmaps_gt = salmap_data.squeeze(-1).numpy().astype(np.uint8)
            eval_result = eval_video_saliency(salmaps_pred, salmaps_gt)  # (B, T, 4)
            all_results.append(eval_result)




    if hasattr(cfg, 'get_salmap') and cfg.get_salmap:
        print('get_salmap exists!')
    else:
        print('get_salmap not exists!')
    cfg.update(get_salmap=True)
    if hasattr(cfg, 'get_salmap') and cfg.get_salmap:
        print('get_salmap exists!')
    else:
        print('get_salmap not exists!')

if __name__ == "__main__":
    
    # input command:
    # python main_test_RLsaliency.py --output output/SAC_AE_GG_v5 --phase test --num_workers 4 --config cfgs/sac_ae_mlnet.yml --gpu_id 0

    # parse input arguments
    cfg = parse_configs()
    # fix random seed 
    set_deterministic(cfg.seed)

    # import transplant
    # matlab = transplant.Matlab(jvm=False, desktop=False)
    # matlab.addpath('metrics/saliency/code_forMetrics')

    test_saliency()