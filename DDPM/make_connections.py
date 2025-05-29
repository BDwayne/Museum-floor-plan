import os
import numpy as np
import cv2
from PIL import Image

# 功能标签映射
COLOR_TO_LABEL = {
    1: "外门",
    2: "门厅",
    3: "中庭",
    4: "临时展厅",
    5: "内门",
    6: "展厅",
    7: "外边",
    8: "过道",
    9: "读书室多功能厅",
    10: "隐形连接",
    11: "其他功能",
    12: "墙"
}

connected_components_labels = [2, 4, 6, 8, 9]  # 相邻功能色块的标签

def find_adjacent_labels(label_image, index_image, mask):
    """
    找到与特定功能块相邻的两个功能块并返回相邻的功能类型和编号。
    """
    h, w = label_image.shape
    adjacent_info = []
    
    # 遍历每个连通区域块，找到其周围的像素
    adjacent_info = []  # 存储相邻的功能序号和房间序号组合
    seen_neighbors = set()  # 用于跟踪已处理的相邻功能和序号

    for i in range(h):
        for j in range(w):
            if mask[i, j]:  # 当前像素是特定功能块
                neighbors = []  # 保存四个方向的相邻像素
                if i > 0: neighbors.append((i-1, j))  # 上方像素
                if i < h-1: neighbors.append((i+1, j))  # 下方像素
                if j > 0: neighbors.append((i, j-1))  # 左侧像素
                if j < w-1: neighbors.append((i, j+1))  # 右侧像素

                # 检查四周相邻功能的标签和编号
                for ni, nj in neighbors:
                    neighbor_label = label_image[ni, nj]
                    if neighbor_label in connected_components_labels:  # 如果是相邻的功能
                        neighbor_index = index_image[ni, nj]
                        neighbor_pair = (neighbor_label, neighbor_index)

                        # 确保该功能序号和房间序号组合尚未被记录
                        if neighbor_pair not in seen_neighbors:
                            adjacent_info.append(neighbor_pair)
                            seen_neighbors.add(neighbor_pair)  # 记录已处理的相邻组合

        # 如果相邻功能有两个不同的组合，则返回这两个
        if len(adjacent_info) == 2:
            return adjacent_info[0], adjacent_info[1]

    # 如果没有找到两个不同的相邻功能块，返回 None
    return None


def process_image(image_name, labels_dir, index_dir, connections_dir):
    """
    处理每一张图像，识别功能块、相邻功能，并保存连接信息。
    """
    # 构建标签图路径，使用前缀 'label_'
    label_image_path = os.path.join(labels_dir, f"label_{image_name}.png")
    label_image = np.array(Image.open(label_image_path))

    # 构建功能房间序号图路径，使用前缀 'index_'
    index_image_path = os.path.join(index_dir, f"index_{image_name}.png")
    index_image = np.array(Image.open(index_image_path))

    # 提取楼层信息（假设数据名最后一个数字代表楼层）
    floor_level = int(image_name.split("-")[-1])

    # 结果保存
    results = []

    # 分别处理内门和隐形连接
    for func_label, connection_type in [(5, 1), (10, 2)]:
        # 查找功能块的连通区域
        mask = (label_image == func_label).astype(np.uint8) * 255
        num_labels, labels_im = cv2.connectedComponents(mask)

        for i in range(1, num_labels):  # 遍历每个连通区域
            component_mask = (labels_im == i)

            # 查找相邻的两个功能块
            adjacent_labels = find_adjacent_labels(label_image, index_image, component_mask)

            # 检查是否找到了两个相邻功能块
            if adjacent_labels is not None:
                (label_1, index_1), (label_2, index_2) = adjacent_labels

                # 记录连接信息
                results.append(f"{label_1} {index_1} {label_2} {index_2} {connection_type} {floor_level}")

    # 将连接信息保存到txt文件中
    connection_file_path = os.path.join(connections_dir, f"connection_{image_name}.txt")
    with open(connection_file_path, "w") as f:
        for line in results:
            f.write(line + "\n")
    print(f"Processed and saved connections for {image_name}")


def process_folder(labels_dir, index_dir, connections_dir):
    """
    处理文件夹中的所有图像，计算连接信息并保存为txt文件。
    """
    # 遍历标签文件夹中的所有图片
    for filename in os.listdir(labels_dir):
        if filename.endswith(".png"):
            # 提取文件名中 '_' 后面的部分作为 image_name
            image_name = os.path.splitext(filename)[0].split('_')[1]
            process_image(image_name, labels_dir, index_dir, connections_dir)


# 文件夹路径
labels_dir = "./autodl-tmp/SZYDDPM/labels/train"    # 建筑功能的单通道标签图
index_dir = "./autodl-tmp/SZYDDPM/index/train"      # 不同功能房间序号的单通道标签图
connections_dir = "./autodl-tmp/SZYDDPM/connections/train"  # 保存连接信息的txt文件夹

# 确保连接文件夹存在
os.makedirs(connections_dir, exist_ok=True)

# 处理文件夹中的所有图像
process_folder(labels_dir, index_dir, connections_dir)
