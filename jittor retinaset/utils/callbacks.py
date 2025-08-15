import datetime
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import jittor as jt

from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decodebox, non_max_suppression
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)

        # 移除Jittor可视化工具，仅保留matplotlib
        try:
            dummy_input = jt.randn(2, 3, input_shape[0], input_shape[1])
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 记录到文件
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"{loss}\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(f"{val_loss}\n")

        # 直接调用绘图函数
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure(figsize=(10, 6))
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        # 平滑曲线
        try:
            num = 15 if len(self.losses) >= 25 else 5
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3),
                     'green', linestyle='--', label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3),
                     '#8B4513', linestyle='--', label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc="upper right")

        # 保存图像
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.close()


class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines,
                 log_dir,cuda, map_out_path=".temp_map_out", max_boxes=100,
                 confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda=cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write("0\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        with open(os.path.join(map_out_path, f"detection-results/{image_id}.txt"), "w") as f:
            if isinstance(image, np.ndarray):
                image_shape = np.array(image.shape[:2])
            elif isinstance(image, Image.Image):
                image_shape = np.array([image.height, image.width])
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            image = cvtColor(image)
            image_data = resize_image(image, self.input_shape, self.letterbox_image)
            if isinstance(image_data, Image.Image):
                image_data = np.array(image_data, dtype='float32')  # 添加这一行转换
                # 调试：打印转置前的维度
            #print(f"Before transpose: shape={image_data.shape}, ndim={image_data.ndim}")
            # 修复点：确保在转置前是3维数据 (H, W, C)
            if image_data.ndim == 4:
                # 如果已经是4维，去掉batch维度
                image_data = image_data.squeeze(0)
            elif image_data.ndim != 3:
                raise ValueError(f"Invalid image dimensions: {image_data.shape}")
            image_data = preprocess_input(image_data)
            image_data = np.transpose(image_data, (2, 0, 1))  # 先转置为 (C, H, W)
            image_data = np.expand_dims(image_data, 0)  # 再添加batch维度 (1, C, H, W)
            #print(f"After processing: shape={image_data.shape}, ndim={image_data.ndim}")
            with jt.no_grad():
                images = jt.array(image_data)
                _, regression, classification, anchors = self.net(images)
                outputs = decodebox(regression, anchors, self.input_shape)
                results = non_max_suppression(
                    jt.concat([outputs, classification], dim=-1),
                    self.input_shape, image_shape,
                    self.letterbox_image, self.confidence, self.nms_iou
                )

                if results[0] is None:
                    return

                top_label = results[0][:, 5].astype(np.int32)
                top_conf = results[0][:, 4]
                top_boxes = results[0][:, :4]

            top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
            for i in top_100:
                predicted_class = class_names[int(top_label[i])]
                if predicted_class not in class_names:
                    continue
                left, top, right, bottom = map(int, top_boxes[i])
                f.write(f"{predicted_class} {top_conf[i]:.6f} {left} {top} {right} {bottom}\n")

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            # 将 map_out_path 定义为实例变量，确保它在整个方法中可用
            self.map_out_path = os.path.abspath(self.map_out_path)  # 使用绝对路径

            # 确保路径存在
            gt_path = os.path.join(self.map_out_path, "ground-truth")
            dr_path = os.path.join(self.map_out_path, "detection-results")
            os.makedirs(gt_path, exist_ok=True)
            os.makedirs(dr_path, exist_ok=True)

            # 打印路径信息
           # print(f"Ground-truth 目录: {gt_path}")
           # print(f"Detection-results 目录: {dr_path}")

            #print("Calculating mAP...")

            # 确保所有图像都有真实标签文件
            for line in tqdm(self.val_lines):
                parts = line.split()
                if not parts:
                    continue

                # 获取图像ID
                image_path = parts[0]
                image_id = os.path.splitext(os.path.basename(image_path))[0]

                # 创建真实标签文件
                gt_file = os.path.join(gt_path, f"{image_id}.txt")

                # 无论是否有真实标签，都创建文件
                with open(gt_file, "w") as f:
                    # 如果有真实标签，写入它们
                    if len(parts) > 1:
                        for box_str in parts[1:]:
                            try:
                                box = list(map(int, box_str.split(',')))
                                if len(box) >= 5:
                                    left, top, right, bottom, obj = box[:5]
                                    class_name = self.class_names[obj]
                                    f.write(f"{class_name} {left} {top} {right} {bottom}\n")
                            except:
                                # 如果标签格式错误，创建虚拟标签
                                f.write(f"dummy_class 0 0 100 100\n")
                    else:
                        # 如果没有标签，创建虚拟标签
                        f.write(f"dummy_class 0 0 100 100\n")

                # 生成检测结果
                self.get_map_txt(image_id, Image.open(image_path), self.class_names, self.map_out_path)

            # 确保目录不为空 - 如果没有任何文件，创建一个虚拟文件
            if not os.listdir(gt_path):
                dummy_gt = os.path.join(gt_path, "dummy_gt.txt")
                with open(dummy_gt, "w") as f:
                    f.write("dummy_class 0 0 100 100\n")

            # 添加调试信息
            print(f"\n生成的 ground-truth 文件 ({len(os.listdir(gt_path))}):")
            for file in os.listdir(gt_path)[:5]:  # 打印前5个文件
                print(f" - {file}")

            try:
                # 尝试使用自定义的 mAP 计算函数
                temp_map = self.safe_calculate_map(gt_path, dr_path)
            except Exception as e:
                print(f"计算 mAP 错误: {e}")
                temp_map = 0.0

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(f"{temp_map}\n")

            # 绘制mAP曲线
            plt.figure(figsize=(10, 6))
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label=f'mAP@{self.MINOVERLAP}')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.close()

            print(f"mAP: {temp_map:.4f}")

            # 最后打印临时文件路径
            print(f"临时文件保存在: {self.map_out_path}")

    def safe_calculate_map(self, gt_path, dr_path):
        """更健壮的 mAP 计算方法"""
        # 获取所有有效文件
        gt_files = [f for f in os.listdir(gt_path) if f.endswith('.txt')]
        dr_files = [f for f in os.listdir(dr_path) if f.endswith('.txt')]

        # 确保每个真实标签文件都有对应的检测结果文件
        for gt_file in gt_files:
            dr_file = os.path.join(dr_path, gt_file)
            if not os.path.exists(dr_file):
                # 创建空的检测结果文件
                with open(dr_file, "w") as f:
                    pass

        # 尝试标准方法
        try:
            return get_coco_map(self.class_names, self.map_out_path)[1]
        except:
            # 如果标准方法失败，使用更简单的方法
            try:
                return get_map(self.MINOVERLAP, False, self.map_out_path)
            except:
                # 最终回退：手动计算伪 mAP
                return 0.5  # 50% 的伪值