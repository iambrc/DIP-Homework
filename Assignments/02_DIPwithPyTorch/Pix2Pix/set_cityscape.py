"""
此脚本用于处理cityscapes数据集
"""

import os
import cv2
import glob


def process_images(input_dir_a, input_dir_b, output_dir):
    for phase in ['train', 'val']:
        input_path_a = os.path.join(input_dir_a, phase)
        input_path_b = os.path.join(input_dir_b, phase)
        output_path = os.path.join(output_dir, phase)
        os.makedirs(output_path, exist_ok=True)

        list_file_path = os.path.join(output_path, f'{phase}_list.txt')
        with open(list_file_path, 'w') as list_file:
            for subdir in os.listdir(input_path_a):
                subdir_path_a = os.path.join(input_path_a, subdir)
                subdir_path_b = os.path.join(input_path_b, subdir)
                output_subdir_path = os.path.join(output_path, subdir)
                os.makedirs(output_subdir_path, exist_ok=True)

                for filename in os.listdir(subdir_path_a):
                    prefix = '_'.join(filename.split('_')[:3])
                    pattern = os.path.join(subdir_path_b, prefix + '*')
                    file_b = glob.glob(pattern)[0]
                    file_a = os.path.join(subdir_path_a, filename)
                    if os.path.exists(file_a) and os.path.exists(file_b):
                        img_a = cv2.imread(file_a)
                        img_b = cv2.imread(file_b)
                        img_a = cv2.resize(img_a, (512, 256))
                        img_b = cv2.resize(img_b, (512, 256))
                        combined_img = cv2.vconcat([img_a, img_b])

                        output_file_path = os.path.join(output_subdir_path, prefix + '.png')
                        cv2.imwrite(output_file_path, combined_img)
                        list_file.write(output_file_path + '\n')


if __name__ == "__main__":
    input_dir_a = 'leftImg8bit_trainvaltest\\leftImg8bit'
    input_dir_b = 'gtFine_trainvaltest\\gtFine'
    output_dir = 'cityscapes'
    process_images(input_dir_a, input_dir_b, output_dir)
