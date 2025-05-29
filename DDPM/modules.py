import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(b, n, h, -1)
        k = self.to_k(context).view(b, context.shape[1], h, -1)
        v = self.to_v(context).view(b, context.shape[1], h, -1)

        q = q.transpose(1, 2)  # [b, h, n, d]
        k = k.transpose(1, 2)  # [b, h, n_ctx, d]
        v = v.transpose(1, 2)  # [b, h, n_ctx, d]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channels,self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(batch_size, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SimpleGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, node_features_1, node_features_2, edge_features, padding_mask):
        # 处理节点特征
        node_emb_1 = self.node_mlp(node_features_1)  # [batch_size, max_rows, output_dim]
        node_emb_2 = self.node_mlp(node_features_2)

        # 处理边特征
        edge_emb = self.edge_mlp(edge_features)      # [batch_size, max_rows, output_dim]

        # 聚合节点和边特征
        combined_emb = node_emb_1 + node_emb_2 + edge_emb  # 简单相加

        # 掩码填充位置
        combined_emb = combined_emb.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 对 max_rows 维度求和，得到每个样本的固定大小嵌入
        structural_emb = combined_emb.sum(dim=1)  # [batch_size, output_dim]

        # 通过最终的 MLP
        structural_emb = self.final_mlp(structural_emb)  # [batch_size, output_dim]

        return structural_emb

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # 输入通道数调整
        self.inc = DoubleConv(c_in, 16)

        # 下采样层
        self.down1 = Down(16, 32)              #128
        #self.sa1 = SelfAttention(32, 128)      
        self.down2 = Down(32, 64)             #64
        #self.sa2 = SelfAttention(64, 64)
        self.down3 = Down(64, 128)             #32
        #self.sa3 = SelfAttention(128, 32)
        self.down4 = Down(128, 256)            #16
        #self.sa4 = SelfAttention(256, 16)
        self.down5 = Down(256, 256)           #8
        #self.sa5 = SelfAttention(256, 8)

        # Bottleneck 部分
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # 上采样层
        self.up1 = Up(512, 128)                #16
        #self.sa6 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)                #32
        #self.sa7 = SelfAttention(64, 32)
        self.up3 = Up(128, 32)                 #64
        #self.sa8 = SelfAttention(32, 64)
        self.up4 = Up(64, 16)                  #128
        #self.sa9 = SelfAttention(16, 128)
        self.up5 = Up(32, 16)                  #256
        #self.sa10 = SelfAttention(16, 256)

        # 输出层
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

         # 定义嵌入层
        num_func_labels = 14        # 功能标签的种类数，根据您的数据调整
        num_room_indices = 20      # 房间编号的可能数量，根据您的数据调整
        num_connection_types = 3    # 连接类型的种类数，根据您的数据调整
        num_floor_levels = 6       # 楼层层数的可能数量，根据您的数据调整
        embedding_dim = 32          # 嵌入维度，可根据需要调整

        self.func_label_embedding = nn.Embedding(num_func_labels, embedding_dim,padding_idx=0)
        self.room_index_embedding = nn.Embedding(num_room_indices, embedding_dim,padding_idx=0)
        self.connection_type_embedding = nn.Embedding(num_connection_types, embedding_dim,padding_idx=0)
        self.floor_level_embedding = nn.Embedding(num_floor_levels, embedding_dim,padding_idx=0)

        # 调整节点和边特征的维度
        node_feature_dim = embedding_dim * 3  # func_label_emb, func_index_emb, floor_level_emb
        edge_feature_dim = embedding_dim * 2  # connection_type_emb, floor_level_emb

        # 定义简单的图神经网络（GNN）
        self.gnn = SimpleGNN(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, output_dim=time_dim)

        # 定义交叉注意力模块
        self.cross_attn = CrossAttention(dim=16, heads=4, dim_head=64)
        # 结构性信息嵌入层（保持不变）
        #self.structural_embedding = nn.Linear(250, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2,device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels //2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels //2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, boundary_image, structural_info):
        # 时间步嵌入和结构性信息嵌入（保持不变）
        t=t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        # 处理结构性信息
        # structural_info 的形状为 [batch_size, max_rows, 6]
        batch_size, max_rows, _ = structural_info.size()
        structural_info = structural_info.to(self.device)

        # 提取各部分信息
        func1_labels = structural_info[:, :, 0]     # [batch_size, max_rows]
        func1_indices = structural_info[:, :, 1]
        func2_labels = structural_info[:, :, 2]
        func2_indices = structural_info[:, :, 3]
        connection_types = structural_info[:, :, 4]
        floor_levels = structural_info[:, :, 5]

        # 标记填充值
        padding_mask = (func1_labels == 0)

        # 获取嵌入表示
        func1_label_emb = self.func_label_embedding(func1_labels)      # [batch_size, max_rows, embedding_dim]
        func1_index_emb = self.room_index_embedding(func1_indices)
        func2_label_emb = self.func_label_embedding(func2_labels)
        func2_index_emb = self.room_index_embedding(func2_indices)
        connection_type_emb = self.connection_type_embedding(connection_types)
        floor_level_emb = self.floor_level_embedding(floor_levels)

        # 构建节点特征，将 floor_level_emb 拼接到节点特征中
        node_features_1 = torch.cat([func1_label_emb, func1_index_emb, floor_level_emb], dim=-1)
        node_features_2 = torch.cat([func2_label_emb, func2_index_emb, floor_level_emb], dim=-1)

        # 构建边特征，将 floor_level_emb 拼接到边特征中
        edge_features = torch.cat([connection_type_emb, floor_level_emb], dim=-1)

        # 使用 GNN 处理结构性信息
        structural_emb = self.gnn(node_features_1, node_features_2, edge_features, padding_mask)
        structural_emb = structural_emb.unsqueeze(1)  # [batch_size, 1, time_dim]

        # 将结构性嵌入与时间嵌入相加
        # t = t + structural_emb

        # 输入拼接
        # x = torch.cat([x, boundary_image], dim=1)

        # 编码器部分
        # 不再将结构性嵌入与时间嵌入相加，而是使用交叉注意力机制
        # 将 structural_emb 扩展维度，以匹配 x1 的形状
        x1 = self.inc(x)  # x1 形状为 [batch_size, 16, H, W]
        channels_x1 = x1.shape[1]  # x1 的通道数
        self.structural_proj = nn.Linear(self.time_dim, channels_x1).to(self.device)  # 定义线性层
        structural_emb = self.structural_proj(structural_emb)  # [batch_size, 1, channels_x1]

        # 对 x1 进行重塑
        batch_size, channels_x1, height, width = x1.shape
        x1_reshaped = x1.permute(0, 2, 3, 1).reshape(batch_size, -1, channels_x1)  # [batch_size, H*W, channels_x1]

        # 应用交叉注意力机制
        x1_attn = self.cross_attn(x1_reshaped, structural_emb)  # [batch_size, H*W, channels_x1]

        # 恢复 x1 的形状
        x1 = x1_attn.reshape(batch_size, height, width, channels_x1).permute(0, 3, 1, 2)  # [batch_size, channels_x1, H, W]

        x2 = self.down1(x1, t)    
        #x2 = self.sa1(x2)         
        x3 = self.down2(x2, t)    
        #x3 = self.sa2(x3)
        x4 = self.down3(x3, t)    
        #x4 = self.sa3(x4)
        x5 = self.down4(x4, t)    
        #x5 = self.sa4(x5)
        x6 = self.down5(x5, t)    
        #x6 = self.sa5(x6)

        # Bottleneck
        x6 = self.bot1(x6)        
        x6 = self.bot2(x6)        
        x6 = self.bot3(x6)        

        # 解码器部分
        x = self.up1(x6, x5, t)   
        #x = self.sa6(x)
        x = self.up2(x, x4, t)    
        #x = self.sa7(x)
        x = self.up3(x, x3, t)    
        #x = self.sa8(x)
        x = self.up4(x, x2, t)    
        #x = self.sa9(x)
        x = self.up5(x, x1, t)    
        #x = self.sa10(x)

        output = self.outc(x)
        output=torch.sigmoid(output)
        return output


# if __name__ == '__main__':
#     net = UNet(c_in=2, c_out=1, device="cpu")
#     print(f"Total parameters: {sum([p.numel() for p in net.parameters()])}")
#     x = torch.randn(3, 1, 256, 256)
#     boundary_image = torch.randn(3, 1, 256, 256)
#     t = x.new_tensor([500] * x.shape[0]).long()
#     structural_info = torch.randn(3, 250)
#     output = net(x, t, boundary_image, structural_info)
#     print(f"Output shape: {output.shape}")
