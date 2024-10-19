import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, epochs):
    """
    绘制训练损失和验证损失

    :param train_losses: List of training losses
    :param val_losses: List of validation losses
    :param epochs: Number of epochs
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    save_path = f'loss/loss_curve_epoch_{epochs}.png'
    plt.savefig(save_path)


if __name__ == '__main__':
    # 示例调用
    train_losses = [0.9, 0.8, 0.7, 0.6, 0.5]  # 训练损失示例数据
    val_losses = [1.0, 0.9, 0.8, 0.75, 0.7]  # 验证损失示例数据
    epochs = 5  # 训练的epoch数量

    plot_loss(train_losses, val_losses, epochs)
