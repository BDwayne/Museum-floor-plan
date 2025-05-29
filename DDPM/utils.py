import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

LABEL_TO_COLOR = {
            1: (0, 255, 24),     # Outer Door
            2: (0, 255, 216),    # Lobby
            3: (0, 12, 255),     # Atrium
            4: (255, 144, 0),    # Temporary Exhibition Hall
            5: (252, 255, 0),    # Inner Door
            6: (255, 0, 0),      # Exhibition Hall
            7: (210, 226, 237),  # Outside the Building
            8: (0, 162, 255),    # Corridor
            9: (174, 255, 0),    # Reading Room / Multi-function Hall
            10: (252, 0, 255),   # Invisible Connection
            11: (150, 0, 255),   # Other Functions
            12: (0, 0, 0),       # Wall
            0: (255, 255, 255),  # Background or unspecified label
        }

def custom_make_grid(images, nrow=8, padding=2, normalize=False):
    """
    自定义实现 torchvision.utils.make_grid 功能。
    
    Args:
    - images (Tensor): 形状为 (B, C, H, W)，其中 B 是批次大小，C 是通道数，H 和 W 分别是高度和宽度。
    - nrow (int): 每行显示的图像数量。
    - padding (int): 图像之间的填充宽度。
    - normalize (bool): 是否归一化到 [0, 1]。
    
    Returns:
    - grid (Tensor): 组合好的网格图像，形状为 (C, grid_height, grid_width)。
    """
    
    B, C, H, W = images.shape
    nrows = (B + nrow - 1) // nrow  # 计算行数
    grid_height = nrows * H + (nrows - 1) * padding
    grid_width = nrow * W + (nrow - 1) * padding
    grid = torch.zeros((C, grid_height, grid_width), dtype=images.dtype)

    # 逐个图像填充
    for idx in range(B):
        row = idx // nrow
        col = idx % nrow
        y = row * (H + padding)
        x = col * (W + padding)
        grid[:, y:y + H, x:x + W] = images[idx]

    if normalize:
        grid = (grid - grid.min()) / (grid.max() - grid.min())

    return grid

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def convert_label_to_color( label_image):
        """
        Convert a single-channel label image to an RGB image using LABEL_TO_COLOR mapping.
        """
        height, width = label_image.shape
        color_image = np.zeros((3, height, width), dtype=np.uint8)
        for label, color in LABEL_TO_COLOR.items():
            mask = (label_image == label)
            color_image[0][mask] = color[0]
            color_image[1][mask] = color[1]
            color_image[2][mask] = color[2]
        return color_image

def save_images(images, path,image_name):
    # 检查输入是否是图像列表，如果是则转换为张量
    if isinstance(images, list):
        images = torch.stack(images)  # 将列表转换为形状为 (B, C, H, W) 的张量
    
    # 检查输入的图像是单通道还是多通道
    if images.shape[1] == 1:  # 单通道（灰度图像）
        grid = custom_make_grid(images, normalize=True)
        ndarr = grid.squeeze().to('cpu').numpy()  # 转换为 (H, W)
        im = Image.fromarray((ndarr * 255).astype(np.uint8))  # 适合 PIL 的灰度图像格式
    else:  # 多通道图像（例如 RGB）
        grid = custom_make_grid(images, normalize=True)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()  # 转换为 (H, W, C) 适合 PIL
        im = Image.fromarray((ndarr * 255).astype(np.uint8))  # 转换为适合 PIL 的图像格式
    path=os.path.join(path, image_name)
    # 保存图像
    im.save(path)

class BuildingDataset(Dataset):
    def __init__(self, boundary_dir, labels_dir, connections_dir, image_size=256):
        self.boundary_dir = boundary_dir
        self.labels_dir = labels_dir
        self.connections_dir = connections_dir
        self.image_size = image_size

        # 只获取指定格式的文件（保持与版本1.2一致）
        self.boundary_images = sorted([f for f in os.listdir(boundary_dir) if f.endswith('.png')])
        self.label_images = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
        self.connection_files = sorted([f for f in os.listdir(connections_dir) if f.endswith('.txt')])

        assert len(self.boundary_images) == len(
            self.label_images) == len(self.connection_files), "Mismatch in dataset lengths"

        # 定义用于处理掩码的变换
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            torchvision.transforms.ToTensor(),  # 转换为 [0,1] 范围的张量
        ])

        # 定义用于处理 RGB 标签图像的变换
        self.label_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),  # 转换为 [0,1] 范围的张量，形状为 [C, H, W]
        ])

    def __len__(self):
        return len(self.boundary_images)

    def __getitem__(self, idx):
        # 加载并处理 boundary_image（单通道掩码）
        boundary_path = os.path.join(self.boundary_dir, self.boundary_images[idx])
        boundary_image = Image.open(boundary_path).convert('L')  # 保持为单通道
        boundary_image = self.mask_transform(boundary_image)
        boundary_image = boundary_image.squeeze(0)  # 移除通道维度，形状为 [H, W]
        boundary_image = boundary_image.long()  # 转换为 long 类型

        # 加载并处理 label_image（RGB 图像）
        label_path = os.path.join(self.labels_dir, self.label_images[idx])
        label_image = Image.open(label_path).convert('RGB')  # 转换为 RGB 模式
        label_image = self.label_transform(label_image)  # 形状为 [3, H, W]
        label_image = label_image.float()  # 确保数据类型为 float

        # 移除通道维度（如果需要）
        # boundary_image = boundary_image.squeeze(0)
        # label_image = label_image.squeeze(0)

        # 加载结构性信息（保持不变）
        connection_path = os.path.join(self.connections_dir, self.connection_files[idx])
        with open(connection_path, 'r') as f:
            lines = f.readlines()
            # 处理结构性信息
            structural_info = []
            for line in lines:
                tokens = line.strip().split()
                if len(tokens) == 6:
                    # [功能1标签, 功能1编号, 功能2标签, 功能2编号, 连接类型, 楼层层数]
                    structural_info.append([int(t) for t in tokens])
            # 转换为张量
            structural_info = torch.tensor(structural_info, dtype=torch.long)
            # 填充或截断至固定大小
            max_rows = 50  # 根据需要调整
            if structural_info.size(0) < max_rows:
                padding = torch.full((max_rows - structural_info.size(0), 6), 0, dtype=torch.long)
                structural_info = torch.cat([structural_info, padding], dim=0)
            else:
                structural_info = structural_info[:max_rows, :]

        return boundary_image, label_image, structural_info


def get_data(args):
    dataset = BuildingDataset(
        boundary_dir=os.path.join(args.dataset_path, 'boundary/train'),
        labels_dir=os.path.join(args.dataset_path, 'dataset/train'),
        connections_dir=os.path.join(args.dataset_path, 'connections/train'),
        image_size=args.image_size
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)