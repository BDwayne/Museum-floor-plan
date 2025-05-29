import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import shutil

# 定义数据增强操作
fill_color = (210, 226, 237)  # 设定填充颜色

transform = transforms.Compose([
    transforms.RandomRotation(30, fill=fill_color),          # 指定填充颜色
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(256, scale=(0.7, 2.8)),
    transforms.RandomAffine(degrees=15, shear=0.2, fill=fill_color),  # 仿射变换指定填充颜色
    transforms.ToTensor(),
])

# 扩展图像数据集并同步扩展连接文件
def augment_images(input_dir, output_dir, connection_dir, connection_output_dir, start_number=143):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(connection_output_dir):
        os.makedirs(connection_output_dir)

    image_filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    current_number = start_number

    for img_name in image_filenames:
        # 提取层数信息
        base_name, floor_level = img_name.split('-')
        floor_level = floor_level.rstrip('.png')

        img_path = os.path.join(input_dir, img_name)
        connection_path = os.path.join(connection_dir, f"connection_{base_name}-{floor_level}.txt")
        
        # 确保连接文件存在
        if not os.path.exists(connection_path):
            print(f"Warning: {connection_path} 不存在，跳过该图像的连接文件复制。")
            continue

        image = Image.open(img_path)

        # 生成一个随机的扩展次数，4、5 或 6
        augment_times = random.choice([4, 5, 6])

        for i in range(augment_times):
            # 图像数据增强
            augmented_image = transform(image)
            augmented_image_pil = transforms.ToPILImage()(augmented_image)  # 转换为PIL图像

            # 设置新的文件名
            new_img_name = f"{current_number}-{floor_level}.png"
            augmented_image_pil.save(os.path.join(output_dir, new_img_name))

            # 复制并重命名连接文件
            new_connection_name = f"connection_{current_number}-{floor_level}.txt"
            shutil.copy(connection_path, os.path.join(connection_output_dir, new_connection_name))

            current_number += 1

        print(f"扩展了 {augment_times} 张图片和连接文件：{img_name} -> {augment_times} 张增强图片")

# 使用数据增强
input_dir = './autodl-tmp/SZYDDPM/dataset/train'  # 输入目录，包含原始图像
output_dir = './autodl-tmp/SZYDDPM/dataset/train'  # 输出目录，用于保存增强后的图像
connection_dir = './autodl-tmp/SZYDDPM/connections/train'  # 输入的连接文件目录
connection_output_dir = './autodl-tmp/SZYDDPM/connections/train'  # 输出的连接文件目录
start_number = 143  # 从143开始的编号

augment_images(input_dir, output_dir, connection_dir, connection_output_dir, start_number)
