import cv2
import numpy as np
import jittor as jt
from PIL import Image
from jittor.dataset import Dataset
import logging
import os
import traceback
from utils.utils import cvtColor, preprocess_input

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认输入尺寸
DEFAULT_SHAPE = (600, 600)


def retinanet_dataset_collate(batch):
    """完全重构的collate函数，强制统一形状"""
    # 找出批次中的最大标注数
    max_boxes = max(len(boxes) for _, boxes in batch) if batch else 0

    images = []
    bboxes = []

    for img, boxes in batch:
        # 确保图像是3通道CHW格式
        if img.ndim == 2:  # 灰度图 [H,W]
            img = jt.stack([img] * 3, dim=0)  # [3,H,W]
        elif img.shape[0] == 1:  # 单通道 [1,H,W]
            img = img.repeat(3, 1, 1)  # [3,H,W]
        elif img.shape[0] > 3:  # 多通道 [C,H,W], C>3
            img = img[:3]  # 取前3通道

        # 强制统一形状为 [3, H, W]
        if img.shape != (3, *DEFAULT_SHAPE):
            # 使用零填充而不是缩放，保持原始比例
            padded_img = jt.zeros((3, *DEFAULT_SHAPE))
            c, h, w = img.shape
            h = min(h, DEFAULT_SHAPE[0])
            w = min(w, DEFAULT_SHAPE[1])
            padded_img[:, :h, :w] = img[:, :h, :w]
            img = padded_img

        images.append(img.float() / 255.0)

        # 处理标注，填充到最大长度
        padded_boxes = jt.zeros((max_boxes, 5))  # 固定5个值
        if len(boxes) > 0:
            padded_boxes[:len(boxes)] = boxes[:, :5]  # 只取前5个值

        bboxes.append(padded_boxes)

    # 堆叠所有图像和标注
    images = jt.stack(images)
    bboxes = jt.stack(bboxes)

    return images, bboxes


class RetinanetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super().__init__()
        self.set_attrs(
            total_len=len(annotation_lines),
            collate_batch=retinanet_dataset_collate  # 单独设置collate函数
        )

        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(annotation_lines)
        # 全局错误统计
        self.error_count = 0
        self.max_errors = 100  # 最多记录错误数

        logger.info(f"Dataset initialized with {len(annotation_lines)} samples")
        logger.info(f"Input shape: {input_shape}, Training mode: {train}")

    def __getitem__(self, index):
        # 确保索引在有效范围内
        index = index % self.length

        try:
            annotation_line = self.annotation_lines[index].strip()
            if not annotation_line:
                logger.error(f"Empty annotation line at index {index}")
                return self._get_placeholder_data()

            # 安全解析行内容
            parts = annotation_line.split()
            if len(parts) < 1:
                logger.error(f"Invalid annotation format at index {index}: '{annotation_line}'")
                return self._get_placeholder_data()

            img_path = parts[0]

            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                logger.error(f"Image file not found: {img_path}")
                return self._get_placeholder_data()

            # 获取图像和边界框
            image, box = self.get_random_data(annotation_line)

            # 验证图像数据
            if image is None or image.size == 0:
                logger.error(f"Empty image data for: {img_path}")
                return self._get_placeholder_data()

            # 转换为CHW格式
            if image.ndim == 3:  # HWC
                image = np.transpose(image, (2, 0, 1))  # 转为CHW

            # 确保数据类型正确
            image = image.astype(np.float32)

            # 转换为Jittor张量
            image_tensor = jt.array(image)
            box_tensor = jt.array(np.array(box, dtype=np.float32)) if len(box) > 0 else jt.zeros((0, 5))

            return image_tensor, box_tensor

        except Exception as e:
            self.error_count += 1
            if self.error_count <= self.max_errors:
                logger.error(f"Error loading sample {index}: {str(e)}")
                logger.debug(traceback.format_exc())

            return self._get_placeholder_data()

    def _get_placeholder_data(self):
        """返回占位符数据，确保形状一致"""
        return jt.zeros((3, *self.input_shape)), jt.zeros((0, 5))

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line):
        """重构的数据获取方法，确保输出形状一致"""
        h, w = self.input_shape
        blank_image = np.zeros((h, w, 3), dtype=np.uint8)
        blank_box = np.zeros((0, 5))

        try:
            parts = annotation_line.split()
            if len(parts) < 1:
                return blank_image, blank_box

            img_path = parts[0]

            # 读取图像
            image = Image.open(img_path)
            image = cvtColor(image)
            iw, ih = image.size

            # 解析边界框
            box = []
            for b_str in parts[1:]:
                b_vals = b_str.split(',')
                if len(b_vals) >= 5:
                    try:
                        box.append([float(x) for x in b_vals[:5]])
                    except ValueError:
                        continue
            box = np.array(box) if box else np.zeros((0, 5))

            # 应用数据增强
            if self.train:
                return self._apply_random_augmentation(image, box, iw, ih)
            else:
                return self._apply_fixed_processing(image, box, iw, ih)

        except Exception as e:
            logger.error(f"Error in get_random_data: {str(e)}")
            return blank_image, blank_box

    def _apply_fixed_processing(self, image, box, orig_w, orig_h):
        """固定处理流程（用于验证）"""
        h, w = self.input_shape

        # 保持长宽比缩放
        scale = min(w / orig_w, h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        dx = (w - new_w) // 2
        dy = (h - new_h) // 2

        # 缩放图像
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 创建新图像并粘贴
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.uint8)

        # 调整边界框
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy

            # 裁剪到图像范围内
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            box[:, 2] = np.clip(box[:, 2], 0, w - 1)
            box[:, 3] = np.clip(box[:, 3], 0, h - 1)

            # 过滤无效框
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            valid_mask = (box_w > 1) & (box_h > 1)
            box = box[valid_mask]

        return image_data, box

    def _apply_random_augmentation(self, image, box, orig_w, orig_h):
        """随机增强处理（用于训练）"""
        h, w = self.input_shape

        # 随机缩放
        scale = self.rand(0.25, 2)
        new_ar = w / h * self.rand(0.7, 1.3)  # 限制长宽比变化范围
        if new_ar < 1:
            new_h = int(scale * h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * w)
            new_h = int(new_w / new_ar)

        # 缩放图像
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 随机位置
        dx = int(self.rand(0, w - new_w))
        dy = int(self.rand(0, h - new_h))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        # 随机颜色调整
        try:
            # 确保3通道
            if image_data.shape[-1] == 4:
                image_data = image_data[:, :, :3]
            elif image_data.shape[-1] == 1:
                image_data = np.stack([image_data[:, :, 0]] * 3, axis=-1)

            # HSV空间调整
            hsv_img = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV).astype(np.float32)

            # 随机扰动
            h_shift = np.random.uniform(-18, 18)  # ±18度
            s_scale = np.random.uniform(0.5, 1.5)
            v_scale = np.random.uniform(0.5, 1.5)

            hsv_img[..., 0] = (hsv_img[..., 0] + h_shift) % 180
            hsv_img[..., 1] = np.clip(hsv_img[..., 1] * s_scale, 0, 255)
            hsv_img[..., 2] = np.clip(hsv_img[..., 2] * v_scale, 0, 255)

            image_data = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except Exception:
            pass  # 如果颜色调整失败，继续使用原图

        # 调整边界框
        if len(box) > 0:
            # 应用位置变换
            box[:, [0, 2]] = box[:, [0, 2]] * (new_w / orig_w) + dx
            box[:, [1, 3]] = box[:, [1, 3]] * (new_h / orig_h) + dy

            # 应用翻转
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]

            # 裁剪到图像范围内
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            box[:, 2] = np.clip(box[:, 2], 0, w - 1)
            box[:, 3] = np.clip(box[:, 3], 0, h - 1)

            # 过滤无效框
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            valid_mask = (box_w > 1) & (box_h > 1)
            box = box[valid_mask]

        # 确保最终尺寸正确
        if image_data.shape[0] != h or image_data.shape[1] != w:
            image_data = cv2.resize(image_data, (w, h))

        return image_data, box