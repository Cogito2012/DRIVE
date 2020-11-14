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
import metrics.saliency.metrics as salmetric
from metrics.eval_tools import evaluation_auc_scores, evaluation_accident_new, evaluate_earliness
from terminaltables import AsciiTable


def setup_dataloader(cfg, num_workers=0, isTraining=False):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]), 'salmap': None,  
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    test_data = DADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict)
    testdata_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    print("# test set: %d"%(len(test_data)))
    return testdata_loader


def eval_video_saliency(pred_salmaps, gt_salmaps, toa_batch=None):
    """Evaluate the saliency maps for a batch of videos"""
    num_videos, num_frames = gt_salmaps.shape[:2]
    metrics_video = np.zeros((num_videos, num_frames, 3), dtype=np.float32)
    for i in range(num_videos):
        for j in range(num_frames):
            map_pred = pred_salmaps[i, j]
            map_gt = gt_salmaps[i, j]
            # We cannot compute AUC metrics (NSS, AUC-Judd, shuffled AUC, and AUC_borji) 
            # since we do not have binary map of human fixation points
            sim = salmetric.SIM(map_pred, map_gt)
            cc = salmetric.CC(map_pred, map_gt)
            kl = salmetric.MIT_KLDiv(map_pred, map_gt)
            metrics_video[i, j, :] = np.array([sim, cc, kl], dtype=np.float32)
    return metrics_video

def test_saliency(fusion, rho=None, margin=None, output_dir=None):

    # prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if fusion == 'dynamic' and margin is not None:
        tag = 'margin%02d'%(int(margin * 100))
        cfg.ENV.fusion_margin = margin
    elif fusion == 'static' and rho is not None:
        tag = 'rho%02d'%(int(rho * 100))
        cfg.ENV.rho = rho
    else:
        print("invalid fusion method %s!"%(fusion))
    result_file = os.path.join(output_dir, 'eval_%s_%s.npy'%(fusion, tag))
    cfg.ENV.use_salmap = True
    cfg.ENV.fusion = fusion

    # initilize environment
    env = DashCamEnv(cfg.ENV, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    # cfg.ENV.output_shape = env.output_shape
    height, width = cfg.ENV.image_shape

    # initialize dataset
    testdata_loader = setup_dataloader(cfg.ENV, cfg.num_workers, isTraining=False)
    # AgentENV
    agent = SAC(cfg.SAC, device=cfg.device)
    # load agent models (by default: the last epoch)
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    agent.load_models(ckpt_dir, cfg)
    agent.set_status('eval')

    if not os.path.exists(result_file):
        all_results = []
        all_pred_scores, all_gt_labels, all_toas = [], [], []
        # start to test 
        with torch.no_grad():
            for i, (video_data, salmap_data, coord_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):  # (B, T, H, W, C)
                # set environment data
                state = env.set_data(video_data, coord_data, data_info)
                # init vars before each episode
                rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                                torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
                score_pred = np.zeros((cfg.ENV.batch_size, env.max_steps), dtype=np.float32)
                salmaps_pred = []
                i_steps = 0
                while i_steps < env.max_steps:
                    salmaps = env.cur_saliency.squeeze(1).cpu().numpy()  # (B, 60, 80)
                    salmaps = np.array([padding_inv(sal, height, width) for sal in salmaps], dtype=np.float32)  # (B, 330, 792)
                    salmaps_pred.append(np.expand_dims(salmaps, axis=1))
                    # select action
                    actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
                    # step
                    state, reward, info = env.step(actions, isTraining=False)
                    score_pred[:, i_steps] = info['pred_score'].cpu().numpy()  # shape=(B,)
                    i_steps += 1
                # evaluate saliency
                salmaps_pred = np.concatenate(salmaps_pred, axis=1)  # (B, T, 330, 792)
                salmaps_gt = salmap_data.squeeze(-1).numpy().astype(np.float32)
                eval_result = eval_video_saliency(salmaps_pred, salmaps_gt)  # (B, T, 3)
                all_results.append(eval_result)
                # gather scores
                all_pred_scores.append(score_pred)  # (B, T)
                all_gt_labels.append(env.clsID.cpu().numpy())  # (B,)
                all_toas.append(env.begin_accident.cpu().numpy())  # (B,)
        # evaluate
        all_pred_scores = np.concatenate(all_pred_scores)
        all_gt_labels = np.concatenate(all_gt_labels)
        all_toas = np.concatenate(all_toas)
        FPS = 30/cfg.ENV.frame_interval
        mTTA = evaluate_earliness(all_pred_scores, all_gt_labels, all_toas, fps=FPS, thresh=0.5)
        AP, p05, r05 = evaluation_accident_new(all_pred_scores, all_gt_labels, all_toas, fps=FPS)
        AUC_video, AUC_frame = evaluation_auc_scores(all_pred_scores, all_gt_labels, all_toas, FPS, video_len=5, pos_only=True, random=False)

        all_results = np.concatenate(all_results, axis=0)  # (N, T, 3)
        saliency_result = np.mean(np.mean(all_results, axis=1), axis=0)
        dict_result = {'SIM': saliency_result[0], 'CC': saliency_result[1], 'KL': saliency_result[2],
                       'TTA': mTTA, 'AP': AP, 'v-AUC': AUC_video, 'f-AUC': AUC_frame}
        np.save(result_file, dict_result)
    else:
        dict_result = np.load(result_file)
    return dict_result


if __name__ == "__main__":
    
    # input command:
    # python main_test_RLsaliency.py --output output/SAC_AE_GG_v5 --phase test --num_workers 4 --config cfgs/sac_ae_mlnet.yml --gpu_id 0

    # parse input arguments
    cfg = parse_configs()
    # fix random seed 
    set_deterministic(cfg.seed)

    output_dir = os.path.join(cfg.output, 'eval-saliency')
    # table head
    metric_names = ['SIM', 'CC', 'KL', 'TTA', 'AP', 'v-AUC', 'f-AUC']
    display_data = [["Metrics"]]
    for name in metric_names:
        display_data[0].append(name)
    
    # evaluate static fusion
    fusion = 'static'
    for idx, rho in enumerate(np.arange(0.0, 1.1, 0.1)):
        items = ["%s, rho=%.1f"%(fusion, rho)]
        print("Process: %s ..."%(items[0]))
        dict_result = test_saliency(fusion, rho=rho, output_dir=output_dir)
        # save results
        for name in metric_names:
            items.append("%.3f"%(dict_result[name]))
        display_data.append(items)

    # evaluate dynamic fusion
    fusion = 'dynamic'
    for margin in np.arange(0.0, 1.1, 0.1):
        items = ["%s, margin=%.1f"%(fusion, margin)]
        print("Process: %s ..."%(items[0]))
        dict_result = test_saliency(fusion, margin=margin, output_dir=output_dir)
        # save results
        for name in metric_names:
            items.append("%.3f"%(dict_result[name]))
        display_data.append(items)

    report_file = os.path.join(output_dir, 'final_report.txt')
    print("All Results Reported in file: \n%s"%(report_file))
    with open(report_file, 'w') as f:
        # print table
        display_title = "Video Saliency Prediction Results on DADA-2000 Dataset."
        table = AsciiTable(display_data, display_title)
        table.inner_heading_row_border = True
        print(table.table)
        f.writelines(table.table)