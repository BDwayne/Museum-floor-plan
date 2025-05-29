import os
import numpy as np
import cv2
from PIL import Image

# 颜色标签映射
COLOR_TO_LABEL = {
    (0, 0, 0): 12,  # 墙
    (0, 255, 24): 1,  # 外门
    (0, 255, 216): 2,  # 门厅
    (0, 12, 255): 3,  # 中庭
    (255, 144, 0): 4,  # 临时展厅
    (252, 255, 0): 5,  # 内门
    (255, 0, 0): 6,  # 展厅
    (210, 226, 237): 7,  # 建筑外部
    (0, 162, 255): 8,  # 过道
    (174, 255, 0): 9,  # 读书室多功能厅
    (252, 0, 255): 10,  # 隐形连接
    (150, 0, 255): 11,  # 其他功能
}

# 需要进行连通区域分析的功能标签
connected_components_labels = [2,4, 6, 8, 9]

# 函数：处理单个图像，计算连通区域并生成通道图
def process_image(image_path, save_path):
    # 读取图像并转换为RGB模式
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 创建一个新的标签图，用于保存不同连通区域的编号
    new_label_image = np.zeros_like(image_np[..., 0])  # 单通道

    # 对每个需要连通区域编号的功能进行连通区域分析
    for color, label in COLOR_TO_LABEL.items():
        if label in connected_components_labels:
            # 找到当前功能颜色的区域
            color_mask = np.all(image_np == color, axis=-1)

            # 将区域掩码转换为 uint8 类型
            color_mask_uint8 = color_mask.astype(np.uint8) * 255

            # 使用 OpenCV 进行连通区域标记
            num_labels, labels_im = cv2.connectedComponents(color_mask_uint8)

            # 对连通区域编号，从 1 开始，不与其他功能编号冲突
            for i in range(1, num_labels):
                new_label_image[labels_im == i] = i  # 各个连通区域编号为 1, 2, 3...

    # 保存生成的单通道图像
    new_image = Image.fromarray(new_label_image)
    new_image.save(save_path)



# 函数：处理文件夹中的所有图像
def process_folder(folder_path, save_folder):
    # 创建保存结果的文件夹
    os.makedirs(save_folder, exist_ok=True)

    # 遍历文件夹中的所有PNG图像
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            
            # 将文件名从 "label_***" 改为 "index_***"
            new_filename = "index_" + filename
            save_path = os.path.join(save_folder, new_filename)

            # 处理图像并保存结果
            process_image(image_path, save_path)
            print(f"Processed and saved {filename}")


# 处理文件夹
input_folder = "./autodl-tmp/SZYDDPM/dataset/train"  # 替换为包含png图像的文件夹路径
output_folder = "./autodl-tmp/SZYDDPM/index/train"  # 替换为保存生成通道图的文件夹路径

process_folder(input_folder, output_folder)
