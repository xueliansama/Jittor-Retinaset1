import unittest
import os
import numpy as np
from PIL import Image
import jittor as jt

# 添加项目根目录到系统路径
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.callbacks import EvalCallback


class TestEvalCallback(unittest.TestCase):
    def setUp(self):
        # 创建测试目录
        self.test_dir = "./test_results"
        os.makedirs(self.test_dir, exist_ok=True)

        # 创建模拟模型
        class MockModel:
            def eval(self):
                return self

            def __call__(self, x):
                # 返回Jittor数组，模拟模型输出
                return [
                    jt.array(np.random.rand(1, 300, 4)),
                    jt.array(np.random.rand(1, 300, 3))
                ]

        self.model = MockModel()
        self.class_names = ["cat", "dog", "bird"]
        self.input_shape = [600, 600]

    def test_pil_image_processing(self):
        """测试处理PIL图像"""
        pil_img = Image.new('RGB', (800, 600), color='blue')

        callback = EvalCallback(
            self.model,
            self.input_shape,
            self.class_names,
            num_classes=3,
            val_lines=["test.jpg"],
            log_dir=self.test_dir,
            cuda=False,
            map_out_path=os.path.join(self.test_dir, "map_out"),
            eval_flag=True,
            period=1
        )

        # 确保目录存在
        detection_dir = os.path.join(self.test_dir, "map_out", "detection-results")
        os.makedirs(detection_dir, exist_ok=True)

        # 调用get_map_txt
        callback.get_map_txt("0", pil_img, self.class_names, os.path.join(self.test_dir, "map_out"))

    def test_numpy_image_processing(self):
        """测试处理numpy图像"""
        np_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        callback = EvalCallback(
            self.model,
            self.input_shape,
            self.class_names,
            num_classes=3,
            val_lines=["test.jpg"],
            log_dir=self.test_dir,
            cuda=False,
            map_out_path=os.path.join(self.test_dir, "map_out"),
            eval_flag=True,
            period=1
        )

        # 确保目录存在
        detection_dir = os.path.join(self.test_dir, "map_out", "detection-results")
        os.makedirs(detection_dir, exist_ok=True)

        # 调用get_map_txt
        callback.get_map_txt("1", np_img, self.class_names, os.path.join(self.test_dir, "map_out"))

    def tearDown(self):
        # 清理测试文件
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
