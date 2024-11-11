import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(Generator, self).__init__()

        # 编码器（下采样）
        self.encoder = nn.ModuleList([
            self.downsample(in_channels, features, normalize=False),  # 不进行批量归一化
            self.downsample(features, features * 2),
            self.downsample(features * 2, features * 4),
            self.downsample(features * 4, features * 8),
            self.downsample(features * 8, features * 8),
            self.downsample(features * 8, features * 8),
            self.downsample(features * 8, features * 8)
        ])

        # 最后一层瓶颈层
        self.bottleneck = self.downsample(features * 8, features * 8, normalize=False)

        # 解码器（上采样）
        self.decoder = nn.ModuleList([
            self.upsample(features * 8, features * 8, dropout=True),
            self.upsample(features * 16, features * 8, dropout=True),
            self.upsample(features * 16, features * 8, dropout=True),
            self.upsample(features * 16, features * 8),
            self.upsample(features * 16, features * 4),
            self.upsample(features * 8, features * 2),
            self.upsample(features * 4, features)
        ])

        # 最后一层输出层
        self.final = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def downsample(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def upsample(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc_outputs = []

        # 编码部分
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码部分
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat([x, enc_outputs[-i - 1]], dim=1)  # 跳跃连接

        return self.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            self.conv_block(in_channels * 2, features, normalize=False),  # 输入是条件图和生成图的拼接
            self.conv_block(features, features * 2),
            self.conv_block(features * 2, features * 4),
            nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1)  # 输出1通道
        )

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        # 将条件图和生成图拼接
        input_img = torch.cat([x, y], dim=1)
        return self.model(input_img)

