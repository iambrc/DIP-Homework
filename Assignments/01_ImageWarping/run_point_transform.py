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


# 执行RBF变换，RBF函数：(r^2+d^2)^alpha
def RBF_deformation(img, source_pts, target_pts, alpha=0.5, r=10.0):
    # 考虑反向映射才能用最后的remap函数，否则最后结果是反过来的
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    height, width = img.shape[:2]
    N = source_pts.shape[0]
    # 构建方程组Mx=y
    y = np.append(target_pts.T.reshape(2 * N), [0, 0, 0, 0, 0, 0]).T
    Px = source_pts[:, 0].reshape(N, 1).repeat(N, axis=1)
    Py = source_pts[:, 1].reshape(N, 1).repeat(N, axis=1)
    Pxy = np.power(np.power(np.subtract(Px, Px.T), 2) + np.power(np.subtract(Py, Py.T), 2) + r ** 2, alpha)
    # 添加约束并构建M
    M = np.zeros([2 * N + 6, 2 * N + 6])
    M[0:N, 0:N] = Pxy
    M[N:2 * N, N:2 * N] = Pxy
    M[0:N, 2 * N:2 * N + 2] = source_pts
    M[N:2 * N, 2 * N + 2:2 * N + 4] = source_pts
    M[2 * N:2 * N + 2,0:N] = source_pts.T
    M[2 * N + 2:2 * N + 4, N:2 * N] = source_pts.T
    M[0:N, 2 * N + 4:2 * N + 5] = np.ones([N, 1])
    M[N:2 * N, 2 * N + 5:2 * N + 6] = np.ones([N, 1])
    M[2 * N + 4:2 * N + 5, 0:N] = np.ones([1, N])
    M[2 * N + 5:2 * N + 6, N:2 * N] = np.ones([1, N])
    # 得到所有的系数
    x = np.linalg.solve(M, y)
    # 计算变形后坐标fv
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pctw, pcth]), 1, 2).T
    fv = np.zeros([height, width, 2])
    # height * width * N * 1
    RBFvalue = np.power(np.linalg.norm(np.subtract(source_pts, img_coordinate.reshape(height, width, 1, 2)), axis=3), 2)
    RBFvalue = np.power(RBFvalue + r ** 2, alpha).reshape(height, width, N, 1)
    coffx = np.repeat(x[0:N].reshape(1, N), width, axis=0)
    coffx = np.repeat(coffx.reshape(1, width, N), height, axis=0).reshape(height, width, 1, N)
    coffy = np.repeat(x[N:2 * N].reshape(1, N), width, axis=0)
    coffy = np.repeat(coffy.reshape(1, width, N), height, axis=0).reshape(height, width, 1, N)
    fv[:, :, 0] = (np.matmul(coffx, RBFvalue).reshape(height, width) + x[2 * N] * img_coordinate[:, :, 0] +
                   x[2 * N + 1] * img_coordinate[:, :, 1] + x[2 * N + 4])
    fv[:, :, 1] = (np.matmul(coffy, RBFvalue).reshape(height, width) + x[2 * N + 2] * img_coordinate[:, :, 0] +
                   x[2 * N + 3] * img_coordinate[:, :, 1] + x[2 * N + 5])
    # 利用opencv remap函数进行重映射
    mapxy = np.float32(fv)
    warped_image = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP,
                             interpolation=cv2.INTER_LINEAR)
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
    qihat = q_hat.reshape(height, width, N, 1, 2)
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
def IDW_deformation(img, source_pts, target_pts, alpha=2.0, eps=1e-8):
    # 考虑反向映射才能用最后的remap函数，否则最后结果是反过来的
    tmp = source_pts
    source_pts = target_pts
    target_pts = tmp

    height, width = img.shape[:2]
    N = source_pts.shape[0]
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pctw, pcth]), 1, 2).T
    # 计算变换的矩阵Di = Ai^(-1) * Bi以及权重Wi
    Pij = np.repeat(source_pts.reshape(N, 1, 2), [N], axis=1).reshape(N, N, 2, 1)
    Pij = np.subtract(Pij, np.swapaxes(Pij, 0, 1))
    Qij = np.repeat(target_pts.reshape(N, 1, 2), [N], axis=1).reshape(N, N, 2, 1)
    Qij = np.subtract(Qij, np.swapaxes(Qij, 0, 1))
    Sigma_ij = np.reciprocal(np.power(np.linalg.norm(Pij, axis=2), alpha) + eps).reshape(N, N)
    np.fill_diagonal(Sigma_ij, 0.0)
    Sigma_ij = np.repeat(Sigma_ij.reshape(N, N, 1, 1), [4], axis=2).reshape(N, N, 2, 2)
    Ai = np.sum(Sigma_ij * np.matmul(Pij, np.swapaxes(Pij, 2, 3)), axis=1).reshape(N, 2, 2)
    Bi = np.sum(Sigma_ij * np.matmul(Qij, np.swapaxes(Pij, 2, 3)), axis=1).reshape(N, 2, 2)
    Di = np.matmul(np.linalg.inv(Ai), Bi)

    Sigma_i = np.reciprocal(np.power(
        np.linalg.norm(np.subtract(source_pts, img_coordinate.reshape(height, width, 1, 2)), axis=3), alpha))
    Wi = np.divide(Sigma_i, np.sum(Sigma_i, axis=2).reshape(height, width, 1)).reshape(height, width, N, 1, 1)
    # 计算相似变形后的坐标fv
    qi = np.repeat(target_pts.reshape(1, 1, N, 2, 1), [height], axis=0)
    qi = np.repeat(qi, [width], axis=1)
    fv_ = np.matmul(Di, np.subtract(img_coordinate.reshape(height, width, 1, 2),
                                    source_pts).reshape(height, width, N, 2, 1)) + qi
    fv = np.sum(np.matmul(fv_, Wi), axis=2).reshape(height, width, 2)
    # 利用opencv remap函数进行重映射
    mapxy = np.float32(fv)
    warped_image = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP,
                             interpolation=cv2.INTER_LINEAR)
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
