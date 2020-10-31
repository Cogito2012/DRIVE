import os
import numpy as np
import torch
from tqdm import tqdm
from src.enviroment import DashCamEnv
from RLlib.SAC.sac import SAC
from main_sac import parse_configs, set_deterministic, setup_dataloader
from metrics.eval_tools import evaluation_fixation, evaluation_auc_scores, evaluation_accident_new, evaluate_earliness


def test_all(testdata_loader, env, agent, test_id=0):
    all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = [], [], [], [], [], []
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
            actions, rnn_state = agent.select_action(state, rnn_state, evaluate=False)  # We use stochastic actions
            # step
            state, reward, info = env.step(actions, isTraining=False)
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


def multi_test():
    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval-multi')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(os.listdir(output_dir)) < 10:
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
            
    mTTA_all, AP_all, p05_all, r05_all, vAUC_all, fAUC_all, mseFix_all = [], [], [], [], [], [], [] 
    for n in range(10):
        result_file = os.path.join(output_dir, 'results_run%d.npz'%(n+1))
        if os.path.exists(result_file):
            save_dict = np.load(result_file, allow_pickle=True)
            all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = \
                save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_fixations'], save_dict['gt_fixations'], save_dict['toas'], save_dict['vids']
        else:
            # start to test 
            with torch.no_grad():
                all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = test_all(testdata_loader, env, agent, test_id=n+1)
            np.savez(result_file[:-4], pred_scores=all_pred_scores, gt_labels=all_gt_labels, pred_fixations=all_pred_fixations, gt_fixations=all_gt_fixations, toas=all_toas, vids=all_vids)

        # evaluate the results
        FPS = 30/cfg.ENV.frame_interval
        mTTA = evaluate_earliness(all_pred_scores, all_gt_labels, all_toas, fps=FPS, thresh=0.5)
        mTTA_all.append(mTTA)

        AP, p05, r05 = evaluation_accident_new(all_pred_scores, all_gt_labels, all_toas, fps=FPS)
        AP_all.append(AP)
        p05_all.append(p05)
        r05_all.append(r05)

        AUC_video, AUC_frame = evaluation_auc_scores(all_pred_scores, all_gt_labels, all_toas, FPS, video_len=5, pos_only=True)
        vAUC_all.append(AUC_video)
        fAUC_all.append(AUC_frame)

        mse_fix = evaluation_fixation(all_pred_fixations, all_gt_fixations)
        mseFix_all.append(mse_fix)
    
    # report the results
    mTTA = np.mean(np.array(mTTA_all))
    std = np.std(np.array(mTTA_all))
    print("\n[Earliness] mTTA@0.5 = %.4f (std=%.4f) seconds."%(mTTA, std))

    AP = np.mean(np.array(AP_all))
    std_ap = np.std(np.array(AP_all))
    p05 = np.mean(np.array(p05_all))
    std_p = np.std(np.array(p05_all))
    r05 = np.mean(np.array(r05_all))
    std_r = np.std(np.array(r05_all))
    print("[Correctness] AP = %.4f (std=%.4f), precision@0.5 = %.4f (std=%.4f), recall@0.5 = %.4f (%.4f)"%(AP, std_ap, p05, std_p, r05, std_r))

    vAUC = np.mean(np.array(vAUC_all))
    std_v = np.std(np.array(vAUC_all))
    fAUC = np.mean(np.array(fAUC_all))
    std_f = np.std(np.array(fAUC_all))
    print("[Correctness] v-AUC = %.5f (std=%.5f), f-AUC = %.5f (std=%.5f)"%(vAUC, std_v, fAUC, std_f))

    MSE = np.mean(np.array(mseFix_all))
    std_mse = np.std(np.array(mseFix_all))
    print('[Attentiveness] Fixation MSE = %.4f (std=%.4f)\n'%(MSE, std_mse))


if __name__ == "__main__":
    
    # input command:
    # CUDA_VISIBLE_DEVICES=$GPU_IDS python main_sac_multitest.py --output ./output/SAC_AE_GG_v5 --phase test --num_workers 0 --config cfgs/sac_ae_mlnet.yml

    # parse input arguments
    cfg = parse_configs()
    # fix random seed 
    set_deterministic(cfg.seed)

    multi_test()
