# test_training.py

import os
import torch
from ddpm import train
from utils import setup_logging

def test_training():
    # 更改当前工作目录到指定路径
    os.chdir('./autodl-tmp/SZYDDPM')
    
    # 设置参数
    class Args:
        run_name = "SZYDDPM_Train"
        epochs = 300  # 仅测试一轮训练
        batch_size = 16  # 根据显存情况调整
        image_size = 256  # 您的数据集图像大小
        dataset_path = "./"  # 数据集路径相对于新的工作目录
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lr = 3e-4

    args = Args()
    setup_logging(args.run_name)

    # 调用 SZYDDPM 中的 train 函数
    train(args)

if __name__ == '__main__':
    test_training()
