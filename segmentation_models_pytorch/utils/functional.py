import torch
import numpy as np


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, class_intervals=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        class_intervals: if None, all class_weights are 1
    Returns:
        float: F score
    """

    if class_intervals is None:
        class_intervals = np.ones((gt.shape[1]), dtype=np.int32)

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Balanced classes
    if 0:
        pr_sampled = torch.zeros(size=(pr.shape[1], 0), device='cuda', dtype=pr.dtype)
        gt_sampled = torch.zeros(size=(gt.shape[1], 0), device='cuda', dtype=gt.dtype)
        for c in range(gt.shape[1]):
            class_indices = (gt[0, c] > 0.5).nonzero()
            sampled_indices = torch.linspace(0, len(class_indices)-1, np.int32(len(class_indices) / class_intervals[c])).long()
            if len(sampled_indices):
                pr_sampled = torch.cat((pr_sampled, pr[0, :, class_indices[sampled_indices,0], class_indices[sampled_indices,1]]), dim=1)
                gt_sampled = torch.cat((gt_sampled, gt[0, :, class_indices[sampled_indices,0], class_indices[sampled_indices,1]]), dim=1)
        pr = pr_sampled
        gt = gt_sampled

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score

def CE(pr, gt, eps=1e-6, threshold=0.5, ignore_channels=None, class_intervals=None):
    """Calculate Cross Entropy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        class_intervals: if None, all class_weights are 1
    Returns:
        float: precision score
    """

    if class_intervals is None:
        class_intervals = np.ones((gt.shape[1]), dtype=np.int32)

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Balanced classes
    if 1:
        score = torch.tensor(0, device='cuda', dtype=pr.dtype)
        samples_num = torch.tensor(0, device='cuda')
        for c in range(gt.shape[1]):
            class_indices = (gt[0, c] > 0.5).nonzero()
            sampled_indices = torch.linspace(0, len(class_indices)-1, np.int32(len(class_indices) / class_intervals[c])).long()
            pr_sampled = pr[0, c, class_indices[sampled_indices,0], class_indices[sampled_indices,1]]
            score += torch.sum( - torch.log(pr_sampled + eps), dtype=pr.dtype)
            samples_num += len(sampled_indices)

        score /= samples_num

    # Unbalanced classes
    else:
        score = torch.sum( - gt * torch.log(pr+eps), dtype=pr.dtype) / (torch.sum(gt) + eps)

    return score



def BCE(pr, gt, eps=1e-6, threshold=0.5, ignore_channels=None):
    """Calculate Binary Cross Entropy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    mask_pos = gt.sum(dim=1) > 0.5

    score = torch.sum(mask_pos * ( (- gt * torch.log(pr+eps)) - ((1-gt) * torch.log((1-pr)+eps)) )) / (torch.sum(mask_pos) + eps)
    return score