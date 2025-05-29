import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[
            :, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, boundary_images, structural_info, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(
                    x, t, boundary_images, structural_info)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                               * predicted_noise) + torch.sqrt(beta) * noise
            
            x = x.clamp(0, 1)  # 假设输出在 [0, 1]

            # 将边界外的像素填充指定颜色
            # boundary_masks = boundary_images.to(self.device).unsqueeze(1).float()  # [n, 1, H, W]
            fill_color = torch.tensor([210/255, 226/255, 237/255], device=self.device).view(1, 3, 1, 1)  # 归一化
            # fill_color_new=fill_color * (1 - boundary_images)
            x = x * boundary_images + fill_color * (1 - boundary_images)

            model.train()
            
            return x  # Return RGB image   

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(c_in=3).to(device)  # 输入通道数为3
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss(reduction='none')
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    loss_history = []

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for i, (boundary_images, labels, structural_info) in enumerate(pbar):
            boundary_images = boundary_images.to(device)  # [batch_size, H, W]
            labels = labels.to(device)  # [batch_size, 3, H, W]
            structural_info = structural_info.to(device)
            t = diffusion.sample_timesteps(labels.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(labels, t)
            predicted_noise = model(x_t, t, boundary_images, structural_info)
            # 计算损失，仅对边界内部的像素计算
            boundary_masks = boundary_images.unsqueeze(1).float()  # [batch_size, 1, H, W]
            loss = mse(noise, predicted_noise)  # [batch_size, 3, H, W]
            loss = loss.mean(dim=1, keepdim=True)  # 对通道维度求平均，形状为 [batch_size, 1, H, W]
            loss = loss * boundary_masks  # 应用掩码
            # loss_sum=loss.sum()
            # boundary_images_sum=boundary_masks.sum()
            loss = loss.sum() / boundary_masks.sum()  # 归一化

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        average_epoch_loss = epoch_loss / l
        loss_history.append(average_epoch_loss)
        logging.info(f"Epoch {epoch} Loss: {average_epoch_loss}")

        # 每10轮保存一次模型和示例图片
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_epoch_{epoch+1}.pt"))

            # 采样并保存图片
            sample_batch = next(iter(dataloader))
            sample_boundary_images = sample_batch[0].to(device).unsqueeze(1).float()
            sample_structural_info = sample_batch[2].to(device)
            sampled_images = diffusion.sample(model, sample_boundary_images, sample_structural_info, n=sample_boundary_images.size(0))
            # 保存采样的图片
            save_images(sampled_images, os.path.join("results", args.run_name),f"epoch_{epoch+1}.png")

    # 绘制并保存损失曲线
    plt.figure()
    plt.plot(range(1, args.epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join("results", args.run_name, "loss_curve.png"))
    plt.close()


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "SZYDDPM"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = "./autodl-tmp/SZYDDPM"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


# if __name__ == '__main__':
#     launch()
