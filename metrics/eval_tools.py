import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline\


def evaluation_accident(all_pred, all_labels, all_toas, all_fps):
    """
    all_pred: list, (N_videos, N_frames) a list of ndarray
    all_labels: list, (N_videos,)
    all_toas: list, (N_videos)
    all_fps: list, (N_videos)
    """
    thresholds = np.arange(0, 1.01, 0.01)
    precisions = np.zeros((len(thresholds)), dtype=np.float32)
    recalls = np.zeros((len(thresholds)), dtype=np.float32)
    mean_toas = np.zeros((len(thresholds)), dtype=np.float32)

    for i, thresh in enumerate(thresholds):
        Ntp, Ntn, Nfp, Nfn = 0, 0, 0, 0
        tta = 0  # time to accident
        Nvid_tp = 0  # number of true positive videos
        # iterate results of each video
        for vid, pred in enumerate(all_pred):
            n_frames = pred.shape[0]
            toa = all_toas[vid]
            fps = all_fps[vid]
            pos = np.sum((pred > thresh).astype(np.int))
            if toa < n_frames and toa >= 0:
                Ntp += pos             # true positive
                Nfn += n_frames - pos  # false negative
                if pos > 0:
                    # time of accident (clipped to larger than 0), unit: second
                    tta += np.maximum(0, toa - np.where(pred > thresh)[0][0]) / fps
                    Nvid_tp += 1
            else:
                Nfp += pos             # false positive
                Ntn += n_frames - pos  # true negative
        
        precisions[i] = Ntp / (Ntp + Nfp) if Ntp + Nfp > 0 else 0
        recalls[i] = Ntp / (Ntp + Nfn) if Ntp + Nfn > 0 else 0
        mean_toas[i] = tta / Nvid_tp if Nvid_tp > 0 else 0

    # sort the results with recall
    inds = np.argsort(recalls)
    precisions = precisions[inds]
    recalls = recalls[inds]
    mean_toas = mean_toas[inds]

    # unique the recall
    new_recalls, indices = np.unique(recalls, return_index=True)
    # for each unique recall, get the best tta and precision
    new_precisions = np.zeros_like(new_recalls)
    new_toas = np.zeros_like(new_recalls)
    for i in range(len(indices)-1):  # first N-1 values
        new_precisions[i] = np.max(precisions[indices[i]:indices[i+1]])
        new_toas[i] = np.max(mean_toas[indices[i]:indices[i+1]])
    new_precisions[-1] = precisions[indices[-1]]
    new_toas[-1] = mean_toas[indices[-1]]

    # compute average precision (AP) score
    AP = 0.0
    if new_recalls[0] != 0:
        AP += new_precisions[0]*(new_recalls[0]-0)
    for i in range(1,len(new_precisions)):
        # compute the area under the P-R curve
        AP += (new_precisions[i-1] + new_precisions[i]) * (new_recalls[i] - new_recalls[i-1]) / 2
    # mean Time to Accident (mTTA)
    mTTA = np.mean(new_toas)
    # TTA at 80% recall
    sort_time = new_toas[np.argsort(new_recalls)]
    sort_recall = np.sort(new_recalls)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))]
    
    return AP, mTTA, TTA_R80



def evaluation_fixation(preds, labels, metric='mse'):
    """Evaluate the Mean Squared Error for fixation prediction
    """
    mse_result = []
    for i, gt_fixes in enumerate(labels):
        inds = np.where(gt_fixes[:, 0] > 0)[0]
        if len(inds) > 0:  # ignore the non-accident frames
            pred_fix = preds[i][inds, :]
            gt_fix = gt_fixes[inds, :]
            mse = np.mean(np.sqrt(np.sum(np.square(pred_fix - gt_fix), axis=1)), axis=0)
            mse_result.append(mse)
    mse_result = np.array(mse_result, dtype=np.float32)
    mse_final = np.mean(mse_result)
    return mse_final


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