import matplotlib.pyplot as plt
import numpy as np


def plot_loss(G_losses, D_losses, epoch, smooth_factor=0.9):
    """
    绘制生成器和判别器的损失曲线，支持平滑处理，并将G和D损失分开绘制。

    :param G_losses: List of Generator losses per batch
    :param D_losses: List of Discriminator losses per batch
    :param epoch: Current epoch number for saving file
    :param smooth_factor: 平滑因子，用于控制损失曲线的平滑度，取值范围在0-1之间
    """

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制生成器损失曲线
    ax1.plot(G_losses, label='Generator Loss', color='blue')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制判别器损失曲线
    ax2.plot(D_losses, label='Discriminator Loss', color='orange')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.set_title('Discriminator Loss')
    ax2.legend()
    ax2.grid(True)

    # 保存图像
    save_path = f'loss/loss_curve_epoch_{epoch}.png'
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curve saved as {save_path}")


if __name__ == '__main__':
    # 示例调用
    G_losses = [0.9, 0.8, 0.7, 0.6, 0.5]  # 训练损失示例数据
    D_losses = [1.0, 0.9, 0.8, 0.75, 0.7]  # 训练损失示例数据
    epochs = 5  # 训练的epoch数量

    plot_loss(G_losses, D_losses, epochs)
