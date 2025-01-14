import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= SE模块 ================= #
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y


# ================= ECA模块 ================= #
class ECABlock(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, 1, channels)
        y = self.conv(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y


# ================= CBAM模块 ================= #
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # 确保 reduction 不超过 in_channels
        self.reduction = max(1, in_channels // reduction)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, self.reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attention[0](x)
        max_out = self.channel_attention[1](x)
        ca = self.channel_attention[2:](avg_out + max_out)
        x = x * ca

        # 空间注意力
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x


# ================= LeNet2 with Attention ================= #
class LeNet2(nn.Module):
    def __init__(self, activation_func, norm_type, attention_type):
        super(LeNet2, self).__init__()
        self.activation_func = activation_func  # 动态传入激活函数
        self.norm_type = norm_type  # 归一化类型
        self.attention_type = attention_type  # 动态传入注意力机制

        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, 3)  # 卷积层1
        self.conv2 = nn.Conv2d(6, 16, 3)  # 卷积层2
        self.conv3 = nn.Conv2d(16, 32, 3)  # 卷积层3
        self.conv4 = nn.Conv2d(32, 64, 3)  # 卷积层4
        self.conv5 = nn.Conv2d(64, 128, 3)  # 卷积层5
        self.conv6 = nn.Conv2d(128, 256, 3)  # 卷积层6

        # 动态归一化层
        self.norm_layers = nn.ModuleList()
        if norm_type == "bn":  # 批归一化
            self.norm_layers = nn.ModuleList([
                nn.BatchNorm2d(6), nn.BatchNorm2d(16),
                nn.BatchNorm2d(32), nn.BatchNorm2d(64),
                nn.BatchNorm2d(128), nn.BatchNorm2d(256)
            ])
        elif norm_type == "ln":  # 层归一化
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm([6, 30, 30]), nn.LayerNorm([16, 28, 28]),
                nn.LayerNorm([32, 12, 12]), nn.LayerNorm([64, 10, 10]),
                nn.LayerNorm([128, 4, 4]), nn.LayerNorm([256, 2, 2])
            ])
        elif norm_type == "gn":  # 组归一化
            self.norm_layers = nn.ModuleList([
                nn.GroupNorm(2, 6), nn.GroupNorm(4, 16),
                nn.GroupNorm(8, 32), nn.GroupNorm(16, 64),
                nn.GroupNorm(32, 128), nn.GroupNorm(32, 256)
            ])

        # 注意力机制模块
        self.attention_blocks = nn.ModuleList()
        if attention_type == "se":
            self.attention_blocks = nn.ModuleList([
                SEBlock(6), SEBlock(16), SEBlock(32),
                SEBlock(64), SEBlock(128), SEBlock(256)
            ])
        elif attention_type == "eca":
            self.attention_blocks = nn.ModuleList([
                ECABlock(6), ECABlock(16), ECABlock(32),
                ECABlock(64), ECABlock(128), ECABlock(256)
            ])
        elif attention_type == "cbam":
            self.attention_blocks = nn.ModuleList([
                CBAMBlock(6, reduction=4),    # 适当减小 reduction
                CBAMBlock(16, reduction=8),
                CBAMBlock(32, reduction=16),
                CBAMBlock(64, reduction=16),
                CBAMBlock(128, reduction=32),
                CBAMBlock(256, reduction=64)
            ])
        else:
            self.attention_blocks = nn.ModuleList([])
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        # Helper function to apply normalization
        def apply_norm(layer_idx, x):
            if self.norm_type is not None:
                return self.norm_layers[layer_idx](x)
            return x

        # Helper function to apply attention
        def apply_attention(layer_idx, x):
            if self.attention_type is not None:
                return self.attention_blocks[layer_idx](x)
            return x

        out = self.conv1(x)
        out = apply_norm(0, out)
        out = apply_attention(0, out)  # 添加注意力模块
        out = self.activation_func(out)

        out = self.conv2(out)
        out = apply_norm(1, out)
        out = apply_attention(1, out)
        out = self.activation_func(out)

        out = F.max_pool2d(out, 2)  # 14x14

        out = self.conv3(out)
        out = apply_norm(2, out)
        out = apply_attention(2, out)
        out = self.activation_func(out)

        out = self.conv4(out)
        out = apply_norm(3, out)
        out = apply_attention(3, out)
        out = self.activation_func(out)

        out = F.max_pool2d(out, 2)  # 5x5

        out = self.conv5(out)
        out = apply_norm(4, out)
        out = apply_attention(4, out)
        out = self.activation_func(out)

        out = self.conv6(out)
        out = apply_norm(5, out)
        out = apply_attention(5, out)
        out = self.activation_func(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation_func(out)
        out = self.fc2(out)
        out = self.activation_func(out)
        out = self.fc3(out)
        return out


