import torch

def exp_loss(pred, target, time, toa):
    '''
    :param pred:
    :param target: onehot codings for binary classification
    :param time:
    :param toa:
    :return:
    '''
    pred = torch.cat([(1.0 - pred).unsqueeze(1), pred.unsqueeze(1)], dim=1)
    # positive example (exp_loss)
    target_cls = target[:, 1]
    target_cls = target_cls.to(torch.long)
    
    penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), toa.to(pred.dtype) - time - 1)
    penalty = torch.where(toa > 0, penalty, torch.zeros_like(penalty).to(pred.device))

    # pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
    nll_loss = torch.nn.NLLLoss(reduction='none')
    pos_loss = -torch.mul(torch.exp(penalty), -nll_loss(torch.log(pred + 1e-6), target_cls))
    # negative example
    # neg_loss = self.ce_loss(pred, target_cls)
    neg_loss = nll_loss(torch.log(pred + 1e-6), target_cls)
    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
    return loss


def fixation_loss(pred_fix, gt_fix, normalize=False, extends=None):
    """r, c
    """
    # Mask out the fixations of accident frames
    mask = gt_fix[:, 0].bool().float() * gt_fix[:, 1].bool().float()  # (B,)
    pred_pts = pred_fix[mask.bool()]
    gt_pts = gt_fix[mask.bool()]
    if pred_pts.size(0) > 0 and gt_pts.size(0) > 0:
        if normalize:
            pred_pts[:, 0] = pred_pts[:, 0] / extends[0]
            pred_pts[:, 1] = pred_pts[:, 1] / extends[1]
            gt_pts[:, 0] = gt_pts[:, 0] / extends[0]
            gt_pts[:, 1] = gt_pts[:, 1] / extends[1]
        # MSE loss
        dist_sq = torch.sum(torch.pow(pred_pts.float() - gt_pts.float(), 2), dim=1, keepdim=True)
        loss = torch.mean(dist_sq)
    else:
        loss = torch.tensor(0.).to(pred_fix.device)
    return loss