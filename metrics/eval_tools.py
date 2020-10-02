import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_auc_score


def draw_pr_curves(precisions, recalls, time_to_accidents, legend_text, vis_file):
    plt.figure(figsize=(10,5))
    fontsize = 18
    plt.plot(recalls, precisions, 'r-')
    plt.axvline(x=0.8, ymax=1.0, linewidth=2.0, color='k', linestyle='--')
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title('Precision Recall Curves', fontsize=fontsize)
    plt.legend([legend_text], fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(vis_file)


def compute_metrics(preds_eval, all_labels, time_of_accidents, thresolds):

    Precision = np.zeros((len(thresolds)))
    Recall = np.zeros((len(thresolds)))
    Time = np.zeros((len(thresolds)))
    cnt = 0
    for Th in thresolds:
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    return Precision, Recall, Time

def evaluation_accident(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    # n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        # n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # compute precision, recall, and tta for each threshold
    thresolds = np.arange(max(min_pred, 0), 1.0, 0.001)
    Precision, Recall, Time = compute_metrics(preds_eval, all_labels, time_of_accidents, thresolds)
    # when threshold=0.5
    p05, r05, t05 = compute_metrics(preds_eval, all_labels, time_of_accidents, [0.5])

    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds

    return AP, mTTA, TTA_R80, p05[0], r05[0], t05[0] * total_seconds


def evaluation_fixation(pred_fixations, gt_fixations, metric='mse'):
    """Evaluate the Mean Squared Error for fixation prediction
    pred_masks: (N, T, 2)
    gt_fixations: (N, T, 2)
    """
    mse_result = []
    for i, (pred_fixes, gt_fixes) in enumerate(zip(pred_fixations, gt_fixations)):
        inds = np.where(gt_fixes[:, 0] > 0)[0]
        if len(inds) > 0:  # ignore the non-accident frames
            pred_fix = pred_fixes[inds, :]
            gt_fix = gt_fixes[inds, :]
            mse = np.mean(np.sqrt(np.sum(np.square(pred_fix - gt_fix), axis=1)), axis=0)
            mse_result.append(mse)
    mse_result = np.array(mse_result, dtype=np.float32)
    mse_final = np.mean(mse_result)
    return mse_final

def evaluation_auc_scores(all_pred_scores, all_gt_labels, all_toas, FPS, video_len=5):
    # compute video-level AUC
    all_vid_scores = [max(pred[:int(toa * FPS)]) for toa, pred in zip(all_toas, all_pred_scores)]
    AUC_video = roc_auc_score(all_gt_labels, all_vid_scores)
    # compute frame-level AUC
    all_frame_scores, all_frame_gts = [], []
    for toa, pred, gt_score in zip(all_toas, all_pred_scores, all_gt_labels):
        toa = video_len if toa <= 0 else toa
        all_frame_scores = np.concatenate((all_frame_scores, pred[:int(toa * FPS)]))
        all_frame_gts = np.concatenate((all_frame_gts, [gt_score] * int(toa * FPS)))
    AUC_frame = roc_auc_score(all_frame_gts, all_frame_scores)
    return AUC_video, AUC_frame


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80, Un in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}, mAU={:.5f}, mEU={:.5f}\n'.format(APvid, AP, mTTA, TTA_R80, Un[0], Un[1]))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    assert vis_batchnum <= len(vis_data)
    for b in range(vis_batchnum):
        results = vis_data[b]
        pred_frames = results['pred_frames']
        labels = results['label']
        toa = results['toa']
        video_ids = results['video_ids']
        detections = results['detections']
        uncertainties = results['pred_uncertain']
        for n in range(batch_size):
            pred_mean = pred_frames[n, :]  # (90,)
            pred_std_alea = 1.0 * np.sqrt(uncertainties[n, :, 0])
            pred_std_epis = 1.0 * np.sqrt(uncertainties[n, :, 1])
            xvals = range(len(pred_mean))
            if smooth:
                # sampling
                xvals = np.linspace(0,len(pred_mean)-1,20)
                pred_mean_reduce = pred_mean[xvals.astype(np.int)]
                pred_std_alea_reduce = pred_std_alea[xvals.astype(np.int)]
                pred_std_epis_reduce = pred_std_epis[xvals.astype(np.int)]
                # smoothing
                xvals_new = np.linspace(1,len(pred_mean)+1,80)
                pred_mean = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)
                pred_std_alea = make_interp_spline(xvals, pred_std_alea_reduce)(xvals_new)
                pred_std_epis = make_interp_spline(xvals, pred_std_epis_reduce)(xvals_new)
                pred_mean[pred_mean >= 1.0] = 1.0-1e-3
                xvals = xvals_new
                # fix invalid values
                indices = np.where(xvals <= toa[n])[0]
                xvals = xvals[indices]
                pred_mean = pred_mean[indices]
                pred_std_alea = pred_std_alea[indices]
                pred_std_epis = pred_std_epis[indices]
            # plot the probability predictions
            fig, ax = plt.subplots(1, figsize=(24, 3.5))
            ax.fill_between(xvals, pred_mean - pred_std_alea, pred_mean + pred_std_alea, facecolor='wheat', alpha=0.5)
            ax.fill_between(xvals, pred_mean - pred_std_epis, pred_mean + pred_std_epis, facecolor='yellow', alpha=0.5)
            plt.plot(xvals, pred_mean, linewidth=3.0)
            if toa[n] <= pred_frames.shape[1]:
                plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            # plt.axhline(y=0.7, xmin=0, xmax=0.9, linewidth=3.0, color='g', linestyle='--')
            # draw accident region
            x = [toa[n], pred_frames.shape[1]]
            y1 = [0, 0]
            y2 = [1, 1]
            ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            fontsize = 25
            plt.ylim(0, 1.1)
            plt.xlim(1, pred_frames.shape[1])
            plt.ylabel('Probability', fontsize=fontsize)
            plt.xlabel('Frame (FPS=20)', fontsize=fontsize)
            plt.xticks(range(0, pred_frames.shape[1], 10), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid(True)
            plt.tight_layout()
            tag = 'pos' if labels[n] > 0 else 'neg'
            plt.savefig(os.path.join(vis_dir, video_ids[n] + '_' + tag + '.png'))
            plt.close()
            # plt.show()