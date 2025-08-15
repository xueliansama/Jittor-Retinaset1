import os
import random
import requests
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import jittor as jt
from PIL import Image


def cvtColor(image: Image.Image) -> Image.Image:
    """将图像转换为RGB格式

    Args:
        image: 输入图像

    Returns:
        转换后的RGB图像
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert('RGB')


def resize_image(image: Image.Image, size: Tuple[int, int],
                 letterbox_image: bool) -> Image.Image:
    """调整图像大小

    Args:
        image: 输入图像
        size: 目标尺寸 (width, height)
        letterbox_image: 是否保持比例填充

    Returns:
        调整后的图像
    """
    iw, ih = image.size
    w, h = size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    return image.resize(size, Image.BICUBIC)


def get_classes(classes_path: str) -> Tuple[List[str], int]:
    with open(classes_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines, len(lines)  # 使用同一个列表


def get_lr(optimizer: 'jittor.optim.Optimizer') -> float:
    """获取当前学习率"""
    return optimizer.lr


def seed_everything(seed: int = 11) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def worker_init_fn(worker_id: int, rank: int, seed: int) -> None:
    """Dataloader工作进程初始化函数"""
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    jt.set_global_seed(worker_seed)


def preprocess_input(image: np.ndarray) -> jt.Var:
    """图像预处理归一化

    Args:
        image: 输入图像数组 (H,W,C)

    Returns:
        归一化后的Jittor变量
    """
    image = image / 255.0
    mean = jt.array([0.406, 0.456, 0.485])
    std = jt.array([0.225, 0.224, 0.229])
    return (image - mean) / std


def show_config(**kwargs: Any) -> None:
    """打印配置信息"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone: str, model_dir: str = "./model_data") -> Optional[str]:
    """下载预训练权重

    Args:
        backbone: 骨干网络编号(0-4)
        model_dir: 模型保存目录

    Returns:
        下载的文件路径，失败返回None
    """
    download_urls = {
        '0': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        '1': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        '2': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        '3': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        '4': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    try:
        url = download_urls[backbone]
        os.makedirs(model_dir, exist_ok=True)

        filename = os.path.join(model_dir, os.path.basename(url))
        if os.path.exists(filename):
            print(f"File {filename} already exists")
            return filename

        print(f"Downloading {url} to {filename}")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename

    except KeyError:
        print(f"Invalid backbone index: {backbone}")
    except Exception as e:
        print(f"Download failed: {str(e)}")
    return None