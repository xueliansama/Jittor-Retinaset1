import numpy as np
import jittor as jt
from jittor import nn
from jittor import misc


def decodebox(regression, anchors, input_shape):
    dtype = regression.dtype
    anchors = anchors.astype(dtype)
    # --------------------------------------#
    #   计算先验框的中心
    # --------------------------------------#
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2

    # --------------------------------------#
    #   计算先验框的宽高
    # --------------------------------------#
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    # --------------------------------------#
    #   计算调整后先验框的宽高
    #   即计算预测框的宽高
    # --------------------------------------#
    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    # --------------------------------------#
    #   计算调整后先验框的中心
    #   即计算预测框的中心
    # --------------------------------------#
    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    # --------------------------------------#
    #   计算预测框的左上角右下角
    # --------------------------------------#
    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    # --------------------------------------#
    #   将预测框的结果进行堆叠
    # --------------------------------------#
    boxes = jt.stack([xmin, ymin, xmax, ymax], dim=2)

    boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]] / input_shape[1]
    boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]] / input_shape[0]

    boxes = jt.clamp(boxes, min_v=0, max_v=1)
    return boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = jt.maximum(b1_x1, b2_x1)
    inter_rect_y1 = jt.maximum(b1_y1, b2_y1)
    inter_rect_x2 = jt.minimum(b1_x2, b2_x2)
    inter_rect_y2 = jt.minimum(b1_y2, b2_y2)

    inter_area = jt.clamp(inter_rect_x2 - inter_rect_x1, min_v=0) * \
                 jt.clamp(inter_rect_y2 - inter_rect_y1, min_v=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / jt.clamp(b1_area + b2_area - inter_area, min_v=1e-6)

    return iou


def retinanet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


from jittor import misc

from jittor import misc


def non_max_suppression(prediction, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # 1. 输入验证
        if image_pred.numel() == 0:
            continue

        # 2. 强制转为2D张量
        if image_pred.ndim == 1:
            image_pred = image_pred.unsqueeze(0)
        elif image_pred.ndim > 2:
            image_pred = image_pred.reshape(-1, image_pred.shape[-1])

        # 3. 安全获取类别信息
        if image_pred.shape[1] < 5:
            continue

        # 新版置信度计算
        obj_conf = image_pred[:, 4]
        class_conf, class_pred = jt.argmax(image_pred[:, 5:], dim=1)
        combined_conf = obj_conf * class_conf

        # 4. 置信度过滤
        conf_mask = (combined_conf >= conf_thres)
        if jt.sum(conf_mask).item() == 0:
            continue

        # 5. 构建检测结果
        detections = jt.contrib.concat([
            image_pred[conf_mask, :4],
            combined_conf[conf_mask].unsqueeze(1),
            class_pred[conf_mask].unsqueeze(1).float()
        ], dim=1)

        # 6. 按类别处理
        unique_labels = jt.unique(detections[:, -1])
        for c in unique_labels:
            # 安全获取类别掩码
            class_mask = (detections[:, -1] == c).reshape(-1)
            if jt.sum(class_mask).item() == 0:
                continue

            # 安全索引
            detections_class = detections[class_mask]

            # 7. 准备NMS输入
            boxes = detections_class[:, :4]
            scores = detections_class[:, 4].reshape(-1)

            if scores.numel() == 0:
                continue

            # 确保维度匹配
            min_len = min(boxes.shape[0], scores.shape[0])
            nms_input = jt.contrib.concat([
                boxes[:min_len],
                scores[:min_len].unsqueeze(1)
            ], dim=1)

            # 8. 执行NMS
            keep = misc.nms(nms_input, nms_thres)
            if keep.numel() == 0:
                continue

            # 累积结果
            if output[i] is None:
                output[i] = detections_class[keep]
            else:
                output[i] = jt.contrib.concat((output[i], detections_class[keep]))

    # 后处理
    for i, out in enumerate(output):
        if out is not None:
            output_np = out.numpy() if isinstance(out, jt.Var) else out
            box_xy = (output_np[:, 0:2] + output_np[:, 2:4]) / 2
            box_wh = output_np[:, 2:4] - output_np[:, 0:2]
            corrected_boxes = retinanet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

            new_output = np.zeros_like(output_np)
            new_output[:, :4] = corrected_boxes
            if output_np.shape[1] > 4:
                new_output[:, 4:] = output_np[:, 4:]
            output[i] = new_output

    return output
