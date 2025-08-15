import datetime
import os
import warnings
from functools import partial
#conda install -c conda-forge libstdcxx-ng=12.3.0 -y
import numpy as np
import jittor as jt
from jittor import nn, optim
from jittor.dataset import Dataset, DataLoader
from tqdm import tqdm

from nets.retinanet import retinanet
from nets.retinanet_training import FocalLoss, get_lr_scheduler, set_optimizer_lr
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import RetinanetDataset  # 移除外部的collate_fn导入
from utils.utils import download_weights, get_classes, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 基本配置
    Cuda = True
    seed = 11
    fp16 = False
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/retinanet_resnet50.pth'
    input_shape = [600, 600]
    phi = 2
    pretrained = False

    # 训练参数
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 4
    UnFreeze_Epoch = 5
    Unfreeze_batch_size = 2
    Freeze_Train = True

    # 优化器参数
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5
    num_workers = 4

    # 数据路径
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # 初始化
    seed_everything(seed)
    jt.flags.use_cuda = 1 if Cuda else 0
    local_rank = 0

    if pretrained:
        download_weights(str(phi))

    # 获取类别和模型
    class_names, num_classes = get_classes(classes_path)
    model = retinanet(num_classes, phi, pretrained, fp16)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model.load(model_path)

    focal_loss = FocalLoss()

    # 日志和回调
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    model_train = model.train()

    # 数据加载
    with open(train_annotation_path) as f:
        all_train_lines = f.readlines()
        # 随机打乱训练集
        np.random.shuffle(all_train_lines)
        # 保留30%的数据 (约400张)
        num_samples = int(len(all_train_lines) * 0.02)
        train_lines = all_train_lines[:num_samples]

    with open(val_annotation_path) as f:
        all_val_lines = f.readlines()
        # 随机打乱验证集
        np.random.shuffle(all_val_lines)
        # 保留30%的数据 (约100张)
        num_samples_val = int(len(all_val_lines) * 0.1)
        val_lines = all_val_lines[:num_samples_val]

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val
        )

    # 训练准备
    UnFreeze_flag = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # 初始epoch_step计算
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size


    if Freeze_Train:
        for param in model.backbone_net.parameters():
            param.stop_grad()

    # 学习率计算
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 优化器设置
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=Init_lr_fit,
            momentum=momentum,
            weight_decay=weight_decay
        )

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    # 数据加载器 - 移除了collate_fn参数
    train_dataset = RetinanetDataset(train_lines, input_shape, num_classes, train=True)
    val_dataset = RetinanetDataset(val_lines, input_shape, num_classes, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )
    map_out_path = os.path.join(log_dir, "map_out")
    os.makedirs(map_out_path, exist_ok=True)

    eval_callback = EvalCallback(
        model, input_shape, class_names, num_classes,
        val_lines, log_dir, Cuda, map_out_path=map_out_path, eval_flag=eval_flag, period=eval_period
    )

    # 训练循环
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            # 重新计算学习率
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            # 解冻 backbone
            for param in model.backbone_net.parameters():
                param.start_grad()

            # 重新设置数据加载器 - 移除了collate_fn参数
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )

            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train=model_train,  # 1. 训练用模型
            model=model,  # 2. 原始模型（用于保存）
            focal_loss=focal_loss,  # 3. 损失函数
            loss_history=loss_history,  # 4. 损失记录器
            eval_callback=eval_callback,  # 5. 评估回调
            optimizer=optimizer,  # 6. 优化器
            epoch=epoch,  # 7. 当前epoch
            epoch_step=epoch_step,  # 8. 每epoch训练步数
            epoch_step_val=epoch_step_val,  # 9. 每epoch验证步数
            gen=train_loader,  # 10. 训练数据加载器
            gen_val=val_loader,  # 11. 验证数据加载器
            Epoch=UnFreeze_Epoch,  # 12. 总epoch数
            cuda=Cuda,  # 13. 是否使用GPU（布尔值）
            save_period=save_period,  # 14. 模型保存间隔
            save_dir=save_dir  # 15. 保存路径
        )

    #loss_history.writer.close()
