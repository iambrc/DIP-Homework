import cv2
import numpy as np
import torch
import sys
from GAN_network import Generator


def load_image(image_path):
    """
    Load and preprocess an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 256))  # 注意调整图片大小
    image = image / 255.0 * 2 - 1  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def save_image(tensor, path):
    """
    Save a tensor as an image.

    Args:
        tensor (torch.Tensor): Image tensor.
        path (str): Path to save the image.
    """
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.cpu().detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))  # Change to (H, W, C)
    tensor = (tensor + 1) / 2 * 255  # Denormalize to [0, 255]
    tensor = tensor.astype(np.uint8)
    # tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, tensor)


def main(image_path, output_path):
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = Generator().to(device)
    # 注意权重选择
    model.load_state_dict(torch.load('generator_epoch_80_cityscapes.pth'))
    model.eval()

    # Load and preprocess the test image
    image = load_image(image_path).to(device)

    # Perform the prediction
    with torch.no_grad():
        output = model(image)

    # Save the output image
    save_image(output, output_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    main(input_image_path, output_image_path)
