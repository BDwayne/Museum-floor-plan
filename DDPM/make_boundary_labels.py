# 定义颜色到类别的映射
COLOR_TO_LABEL = {
    (210, 226, 237): 0,       # 建筑外部
}

from PIL import Image
import numpy as np
import os

# 文件夹路径
image_dir = './autodl-tmp/SZYDDPM/dataset/train'
label_save_dir = './autodl-tmp/SZYDDPM/boundary/train'  # 保存标签图的文件夹

if not os.path.exists(label_save_dir):
    os.makedirs(label_save_dir)

def color_to_label(image_path, color_to_label_map):
    # 打开图片并转换为 RGB 格式
    image = Image.open(image_path).convert("RGB")
    # image = Image.open(image_path)
    image_np = np.array(image)

    # 创建与原图相同大小的空白标签图
    label_image = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)*255

    # 遍历每个像素，将颜色映射为对应的类别
    for color, label in color_to_label_map.items():
        mask = np.all(image_np == color, axis=-1)  # 找到所有该颜色的像素
        label_image[mask] = label

    return label_image

# 遍历所有图像文件
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)

        # 将彩色图像转换为标签图
        label_image = color_to_label(image_path, COLOR_TO_LABEL)

        # 保存标签图
        label_image_pil = Image.fromarray(label_image)
        label_image_pil.save(os.path.join(label_save_dir, f"boundary_{filename}"))

        print(f"Processed {filename} -> Saved boundary image.")
