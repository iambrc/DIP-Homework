import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FacadesDataset, CityscapesDataset
from GAN_network import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from plot_loss import plot_loss


def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])
        comparison = np.vstack((input_img_np, target_img_np, output_img_np))
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


def train_one_epoch(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion_GAN, criterion_L1,
                    device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_preds = discriminator(image_semantic, image_rgb)
        d_loss_real = criterion_GAN(real_preds, torch.ones_like(real_preds))

        fake_images = generator(image_semantic)
        fake_preds = discriminator(image_semantic, fake_images.detach())
        d_loss_fake = criterion_GAN(fake_preds, torch.zeros_like(fake_preds))

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        fake_preds = discriminator(image_semantic, fake_images)
        g_loss_GAN = criterion_GAN(fake_preds, torch.ones_like(fake_preds))
        g_loss_L1 = criterion_L1(fake_images, image_rgb)
        g_loss = g_loss_GAN + g_loss_L1 * 10
        g_loss.backward()
        optimizer_g.step()

        running_loss_g += g_loss.item()
        running_loss_d += d_loss.item()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Generator Loss: {g_loss.item():.4f}, '
            f'Discriminator Loss: {d_loss.item():.4f}')

        if epoch % 5 == 0 and i == 0:
            save_images(image_semantic, image_rgb, fake_images, 'train_results', epoch)

    avg_loss_g = running_loss_g / len(dataloader)
    avg_loss_d = running_loss_d / len(dataloader)
    return avg_loss_g, avg_loss_d


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = CityscapesDataset(list_file='train_list.txt')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)

    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=3).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_g = StepLR(optimizer_g, step_size=50, gamma=0.2)
    scheduler_d = StepLR(optimizer_d, step_size=50, gamma=0.2)

    num_epochs = 800
    train_losses_g = []
    train_losses_d = []

    for epoch in range(num_epochs):
        train_loss_g, train_loss_d = train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d,
                                                     criterion_GAN, criterion_L1, device, epoch, num_epochs)

        train_losses_g.append(train_loss_g)
        train_losses_d.append(train_loss_d)

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 10 == 0:
            plot_loss(train_losses_g, train_losses_d, epoch + 1)

        if (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
