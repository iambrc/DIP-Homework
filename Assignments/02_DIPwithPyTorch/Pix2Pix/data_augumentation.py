import cv2
import os
import random
import numpy as np


def load_and_split_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                h, w = img.shape[:2]
                left_half = img[:, :w//2]
                right_half = img[:, w//2:]
                images.append((filename, left_half, right_half))
    return images


def transform_images(images):
    transformed_images = []
    for filename, left, right in images:
        h, w = left.shape[:2]
        flip_code = random.choice([-1, 0, 1])
        scale_range = (0.9, 1.2)
        scale = random.uniform(*scale_range)
        angle_range = (-30, 30)
        angle = random.uniform(*angle_range)
        M_rotation = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        M_scaling = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)

        transformed_left = cv2.flip(left, flip_code)
        transformed_right = cv2.flip(right, flip_code)
        transformed_left = cv2.warpAffine(transformed_left, M_rotation, (w, h))
        transformed_right = cv2.warpAffine(transformed_right, M_rotation, (w, h))
        transformed_left = cv2.warpAffine(transformed_left, M_scaling, (w, h))
        transformed_right = cv2.warpAffine(transformed_right, M_scaling, (w, h))

        combined_img = np.hstack((transformed_left, transformed_right))
        transformed_images.append((filename, combined_img))
    return transformed_images


def save_transformed_images(images, directory, increment):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename, image in images:
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{int(base_name) + increment}{ext}"
        cv2.imwrite(os.path.join(directory, new_filename), image)


def update_list_file(file_path, new_images, directory, increment):
    with open(file_path, 'a') as file:
        for filename, _ in new_images:
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{int(base_name) + increment}{ext}"
            full_path = os.path.join(directory, new_filename).replace("\\", "/")
            file.write(f"./{full_path}\n")


def transform_and_save(images, output_dir, list_file, increment):
    transformed_images = transform_images(images)
    save_transformed_images(transformed_images, output_dir, increment)
    update_list_file(list_file, transformed_images, output_dir, increment)


if __name__ == '__main__':
    train_images = load_and_split_images('datasets/facades/train')
    val_images = load_and_split_images('datasets/facades/val')

    transform_and_save(train_images, 'datasets/facades/train', 'train_list.txt', 400)
    transform_and_save(val_images, 'datasets/facades/val', 'val_list.txt', 100)
