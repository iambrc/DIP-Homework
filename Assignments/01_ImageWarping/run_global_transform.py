import gradio as gr
import cv2
import numpy as np


# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3), dtype=np.uint8) + np.array(
        (255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    transformed_image = np.array(image_new)
    image_new[pad_size:pad_size + image.shape[0], pad_size:pad_size + image.shape[1]] = image
    image = np.array(image_new)
    # transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    c = np.cos(rotation * np.pi / 180)
    s = np.sin(rotation * np.pi / 180)
    height = image.shape[0]
    width = image.shape[1]

    # 剔除部分区域，减少计算量，可以调用np.dot函数做矩阵乘法，也可以直接按照下面的计算，个人认为比多次调用dot函数和to_3x3函数高效
    new_size = int(scale * np.sqrt((height - 2 * pad_size) ** 2 + (width - 2 * pad_size) ** 2))
    left = max(0, int(translation_x + width / 2 - new_size / 2))
    right = min(width, int(translation_x + width / 2 + new_size / 2))
    up = max(0, int(translation_y + height / 2 - new_size / 2))
    bottom = min(height, int(translation_y + height / 2 + new_size / 2))

    for i in range(up, bottom, 1):
        for j in range(left, right, 1):
            tmp1 = j - width / 2 - translation_x
            tmp2 = i - height / 2 - translation_y
            jj = c / scale * tmp1 + s / scale * tmp2
            ii = -s / scale * tmp1 + c / scale * tmp2
            if flip_horizontal:
                jj = -jj
            jj = jj + width / 2
            ii = ii + height / 2
            if 0 <= ii < height and 0 <= jj < width:
                transformed_image[i, j, :] = image[int(ii), int(jj), :]

    return transformed_image


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


# Launch the Gradio interface
interactive_transform().launch()
