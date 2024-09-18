import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img


# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 红色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 蓝色表示目标点

    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射

    return marked_image


# 执行MLS仿射变换
def MLS_affine_deformation(img, source_pts, target_pts, alpha=1.0, eps=1e-8):
    # 考虑反向映射才能用最后的remap函数，否则最后结果是反过来的
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    height, width = img.shape[:2]
    N = source_pts.shape[0]
    ##  使用numpy的函数快速计算避免使用效率较低的for循环
    # 生成图像坐标矩阵 image_coordinate[i, j] = [j, i] 规模为 height * width * 2
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pctw, pcth]), 1, 2).T
    # 计算w_i
    wi = np.reciprocal(np.power(
        np.linalg.norm(np.subtract(source_pts, img_coordinate.reshape(height, width, 1, 2)) + eps, axis=3), 2 * alpha))
    # 计算p_star, q_star
    p_star = np.divide(np.matmul(wi, source_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    q_star = np.divide(np.matmul(wi, target_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    # 计算p_hat, q_hat : 二者为height * width * N * 2的矩阵，每个点都存着N * 2规模自己对应的p_hat, q_hat
    p_hat = np.subtract(source_pts, p_star.reshape(height, width, 1, 2))
    q_hat = np.subtract(target_pts, q_star.reshape(height, width, 1, 2))
    # 计算仿射变形的矩阵A,B 规模为height * width * N * 2 * 2
    # 先重构wi的形状
    wii = np.repeat(wi.reshape(height, width, N, 1), [4], axis=3)
    pihat = p_hat.reshape(height, width, N, 1, 2)
    pihatT = np.swapaxes(pihat, 3, 4)
    qihat = q_hat.reshape(height, width, N, 1, 2)
    A = (wii * np.matmul(pihatT, pihat).reshape(height, width, N, 4)).reshape(height, width, N, 2, 2)
    B = (wii * np.matmul(pihatT, qihat).reshape(height, width, N, 4)).reshape(height, width, N, 2, 2)
    # 计算仿射变形后的坐标fv 规模为height * width * 2，每个点存着变形后的坐标
    fv = np.matmul((img_coordinate - p_star).reshape(height, width, 1, 2),
                   np.matmul(np.linalg.inv(np.sum(A, axis=2)), np.sum(B, axis=2))).reshape(height, width, 2) + q_star
    # 利用opencv remap函数进行重映射
    mapxy = np.float32(fv)
    warped_image = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP,
                             interpolation=cv2.INTER_LINEAR)
    return warped_image


# 执行RBF变换
def RBF_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    warped_image = np.array(image)
    return warped_image


# 执行MLS相似变换
def MLS_similarity_deformation(img, source_pts, target_pts, alpha=1.0, eps=1e-8):
    # 考虑反向映射才能用最后的remap函数，否则最后结果是反过来的
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    height, width = img.shape[:2]
    N = source_pts.shape[0]
    ##  使用numpy的函数快速计算避免使用效率较低的for循环
    # 生成图像坐标矩阵 image_coordinate[i, j] = [j, i] 规模为 height * width * 2
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pctw, pcth]), 1, 2).T
    # 计算w_i
    wi = np.reciprocal(np.power(
        np.linalg.norm(np.subtract(source_pts, img_coordinate.reshape(height, width, 1, 2)) + eps, axis=3), 2 * alpha))
    # 计算p_star, q_star
    p_star = np.divide(np.matmul(wi, source_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    q_star = np.divide(np.matmul(wi, target_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    # 计算p_hat, q_hat : 二者为height * width * N * 2的矩阵，每个点都存着N * 2规模自己对应的p_hat, q_hat
    p_hat = np.subtract(source_pts, p_star.reshape(height, width, 1, 2))
    q_hat = np.subtract(target_pts, q_star.reshape(height, width, 1, 2))
    # 计算miu_s : height * width * 1
    pihat = p_hat.reshape(height, width, N, 1, 2)
    pihatT = np.swapaxes(pihat, 3, 4)
    qihat = q_hat.reshape(height, width, N, 1, 2)
    miu_s = np.sum(wi.reshape(height, width, N, 1) * np.matmul(pihat, pihatT).reshape(height, width, N, 1),
                   axis=2).reshape(height, width, 1)
    miu_s = np.repeat(miu_s, 2, axis=2)
    # 计算矩阵Ai : height * width * N * 2 * 2
    l1 = pihat
    l2 = np.concatenate((l1[:,:,:,:,1], -l1[:,:,:,:,0]),axis=3).reshape(height, width, N, 1, 2)
    r1 = np.subtract(img_coordinate, p_star)
    r2 = np.repeat(np.swapaxes(np.array([r1[:, :, 1], -r1[:, :, 0]]), 1, 2).T.reshape(height, width, 1, 2, 1),
                   [N], axis=2)
    r1 = np.repeat(r1.reshape(height, width, 1, 2, 1), [N], axis=2)
    wii = np.repeat(wi.reshape(height, width, N, 1), [4], axis=3)
    Ai = (wii * np.concatenate((np.matmul(l1, r1), np.matmul(l1, r2),
        np.matmul(l2, r1), np.matmul(l2, r2)), axis=3).reshape(height, width, N, 4)).reshape(height, width, N, 2, 2)
    # 计算相似变形后的坐标fv
    fv = np.divide((np.sum(np.matmul(qihat, Ai), axis=2)).reshape(height, width, 2), miu_s) + q_star
    # 利用opencv remap函数进行重映射
    mapxy = np.float32(fv)
    warped_image = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP,
                             interpolation=cv2.INTER_LINEAR)
    return warped_image


# 执行MLS刚性变换
def MLS_rigid_deformation(img, source_pts, target_pts, alpha=1.0, eps=1e-8):
    # 考虑反向映射才能用最后的remap函数，否则最后结果是反过来的
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    height, width = img.shape[:2]
    N = source_pts.shape[0]
    ##  使用numpy的函数快速计算避免使用效率较低的for循环
    # 生成图像坐标矩阵 image_coordinate[i, j] = [j, i] 规模为 height * width * 2
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pctw, pcth]), 1, 2).T
    # 计算w_i
    wi = np.reciprocal(np.power(
        np.linalg.norm(np.subtract(source_pts, img_coordinate.reshape(height, width, 1, 2)) + eps, axis=3), 2 * alpha))
    # 计算p_star, q_star
    p_star = np.divide(np.matmul(wi, source_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    q_star = np.divide(np.matmul(wi, target_pts), np.sum(wi, axis=2).reshape(height, width, 1))
    # 计算p_hat, q_hat : 二者为height * width * N * 2的矩阵，每个点都存着N * 2规模自己对应的p_hat, q_hat
    p_hat = np.subtract(source_pts, p_star.reshape(height, width, 1, 2))
    q_hat = np.subtract(target_pts, q_star.reshape(height, width, 1, 2))
    # 计算miu_s : height * width * 1
    pihat = p_hat.reshape(height, width, N, 1, 2)
    pihatT = np.swapaxes(pihat, 3, 4)
    qihat = q_hat.reshape(height, width, N, 1, 2)
    miu_s = np.sum(wi.reshape(height, width, N, 1) * np.matmul(pihat, pihatT).reshape(height, width, N, 1),
                   axis=2).reshape(height, width, 1)
    # 计算矩阵Ai : height * width * N * 2 * 2
    l1 = pihat
    l2 = np.concatenate((l1[:, :, :, :, 1], -l1[:, :, :, :, 0]), axis=3).reshape(height, width, N, 1, 2)
    r1 = np.subtract(img_coordinate, p_star)
    r2 = np.repeat(np.swapaxes(np.array([r1[:, :, 1], -r1[:, :, 0]]), 1, 2).T.reshape(height, width, 1, 2, 1),
                   [N], axis=2)
    r1 = np.repeat(r1.reshape(height, width, 1, 2, 1), [N], axis=2)
    wii = np.repeat(wi.reshape(height, width, N, 1), [4], axis=3)
    Ai = (wii * np.concatenate((np.matmul(l1, r1), np.matmul(l1, r2),
                                np.matmul(l2, r1), np.matmul(l2, r2)), axis=3).reshape(height, width, N, 4)).reshape(
        height, width, N, 2, 2)
    # 计算相似变形后的坐标fv
    fv_ = np.sum(np.matmul(qihat, Ai), axis=2)  # height * width * 1 * 2
    fv = np.linalg.norm(r1[:, :, 0, :, :], axis=2) / (np.linalg.norm(fv_, axis=3) + eps) * fv_[:, :, 0, :] + q_star
    # 利用opencv remap函数进行重映射
    mapxy = np.float32(fv)
    warped_image = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP,
                             interpolation=cv2.INTER_LINEAR)
    return warped_image


# 执行IDW变换
def IDW_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    warped_image = np.array(image)
    return warped_image


def run_warping(warp_type: str):
    global points_src, points_dst, image  ### fetch global variables

    if warp_type == "RBF变换":
        warped_image = RBF_deformation(image, np.array(points_src), np.array(points_dst))
    elif warp_type == "MLS仿射变换":
        warped_image = MLS_affine_deformation(image, np.array(points_src), np.array(points_dst))
    elif warp_type == "MLS相似变换":
        warped_image = MLS_similarity_deformation(image, np.array(points_src), np.array(points_dst))
    elif warp_type == "MLS刚性变换":
        warped_image = MLS_rigid_deformation(image, np.array(points_src), np.array(points_dst))
    elif warp_type == "IDW变换":
        warped_image = IDW_deformation(image, np.array(points_src), np.array(points_dst))
    else:
        return image

    return warped_image


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            warping_type = gr.Radio(label="选择变换类型", choices=["RBF变换", "MLS仿射变换", "MLS相似变换", "MLS刚性变换", "IDW变换"])

        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)

    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, warping_type, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

# 启动 Gradio 应用
demo.launch()
