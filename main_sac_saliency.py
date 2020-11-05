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
from src.data_transform import ProcessImages, ProcessFixations


def setup_dataloader(cfg, num_workers=0):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]),
                      'salmap': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}

    test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=cfg.use_salmap)
    testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    print("# test set: %d"%(len(test_data)))


def test_saliency():

    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval-saliency')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize environment
    env = DashCamEnv(cfg.ENV, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    cfg.ENV.output_shape = env.output_shape
    # initialize dataset
    testdata_loader = setup_dataloader(cfg.ENV, 0, isTraining=False)
    # AgentENV
    agent = SAC(cfg.SAC, device=cfg.device)
    # load agent models (by default: the last epoch)
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    agent.load_models(ckpt_dir, cfg)
    agent.set_status('eval')

    # start to test 
    with torch.no_grad():
        for i, (video_data, _, coord_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader), 
                                                                                    desc='Test: %d / %d'%(test_id, 10)):  # (B, T, H, W, C)
            # set environment data
            state = env.set_data(video_data, coord_data, data_info)

            # init vars before each episode
            rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                            torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
            score_pred = np.zeros((cfg.ENV.batch_size, env.max_steps), dtype=np.float32)
            fixation_pred = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
            fixation_gt = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
            i_steps = 0
            while i_steps < env.max_steps:
                # select action
                actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
                # step
                state, reward, info = env.step(actions, isTraining=False)

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
    # python main_sac_statictest.py --output output/SAC_AE_GG_v5 --phase test --num_workers 4 --config cfgs/sac_ae_mlnet.yml --gpu_id 0

    # parse input arguments
    cfg = parse_configs()
    # fix random seed 
    set_deterministic(cfg.seed)

    cfg.use_salmap = True


    test_saliency()