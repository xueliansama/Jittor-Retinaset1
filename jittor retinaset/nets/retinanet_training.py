import math
from functools import partial

import jittor as jt
from jittor import nn


def calc_iou(a, b):
    max_length = jt.max(a)
    a = a / max_length
    b = b / max_length

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = jt.minimum(a[:, 3].unsqueeze(1), b[:, 2]) - jt.maximum(a[:, 1].unsqueeze(1), b[:, 0])
    ih = jt.minimum(a[:, 2].unsqueeze(1), b[:, 3]) - jt.maximum(a[:, 0].unsqueeze(1), b[:, 1])
    iw = jt.clamp(iw, min_v=0)
    ih = jt.clamp(ih, min_v=0)
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).unsqueeze(1) + area - iw * ih
    ua = jt.clamp(ua, min_v=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def get_target(anchor, bbox_annotation, classification, cuda):
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

    IoU_max, IoU_argmax = jt.argmax(IoU, dim=1)

    targets = jt.ones_like(classification) * -1
    targets = targets.type_as(classification)

    targets[IoU_max < 0.4, :] = 0

    positive_indices = jt.greater_equal(IoU_max, 0.5)

    assigned_annotations = bbox_annotation[IoU_argmax, :]

    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations


def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    assigned_annotations = assigned_annotations[positive_indices, :]

    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = jt.clamp(gt_widths, min_v=1)
    gt_heights = jt.clamp(gt_heights, min_v=1)

    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = jt.log(gt_widths / anchor_widths_pi)
    targets_dh = jt.log(gt_heights / anchor_heights_pi)

    targets = jt.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.transpose(1, 0)
    return targets


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def execute(self, classifications, regressions, anchors, annotations, alpha=0.25, gamma=2.0, cuda=True):
        batch_size = classifications.shape[0]

        dtype = regressions.dtype
        anchor = anchors[0, :, :].astype(dtype)

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []

        for j in range(batch_size):
            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            classification = jt.clamp(classification, 5e-4, 1.0 - 5e-4)

            if len(bbox_annotation) == 0:
                alpha_factor = jt.ones_like(classification) * alpha
                alpha_factor = alpha_factor.astype(classification.dtype)

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * jt.pow(focal_weight, gamma)

                bce = - (jt.log(1.0 - classification))

                cls_loss = focal_weight * bce

                classification_losses.append(cls_loss.sum())
                regression_losses.append(jt.array(0).astype(classification.dtype))
                continue

            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(
                anchor, bbox_annotation, classification, cuda)

            alpha_factor = jt.ones_like(targets) * alpha
            alpha_factor = alpha_factor.astype(classification.dtype)

            alpha_factor = jt.where(jt.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = jt.where(jt.equal(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * jt.pow(focal_weight, gamma)

            bce = - (targets * jt.log(classification) + (1.0 - targets) * jt.log(1.0 - classification))
            cls_loss = focal_weight * bce

            zeros = jt.zeros_like(cls_loss)
            zeros = zeros.astype(cls_loss.dtype)
            cls_loss = jt.where(jt.not_equal(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / jt.clamp(num_positive_anchors.astype(dtype), min_v=1.0))

            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices,
                                      anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)

                regression_diff = jt.abs(targets - regression[positive_indices, :])
                regression_loss = jt.where(
                    regression_diff <= 1.0 / 9.0,
                    0.5 * 9.0 * jt.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(jt.array(0).astype(classification.dtype))

        c_loss = jt.stack(classification_losses).mean()
        r_loss = jt.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05,
                     warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr