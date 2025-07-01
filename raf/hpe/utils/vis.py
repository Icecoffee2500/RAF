# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import random
import matplotlib.pyplot as plt

from hpe.utils.post_processing import get_max_preds

coco_info = {
    "K_NAMES": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "N_KEYPOINTS": 17,
    "IMG_WIDTH": 192,
    "IMG_HEIGHT": 256,
}

skeleton_connection_info = {
    "SKELETON" : np.array([
       [15, 13],
       [13, 11],
       [16, 14],
       [14, 12],
       [11, 12],
       [ 5, 11],
       [ 6, 12],
       [ 5,  6],
       [ 5,  7],
       [ 6,  8],
       [ 7,  9],
       [ 8, 10],
       [ 1,  2],
       [ 0,  1],
       [ 0,  2],
       [ 1,  3],
       [ 2,  4],
       [ 3,  5],
       [ 4,  6]]),

    "COLOR" : np.array([ 
        [0,0,0.8], # left_ankle left_knee
        [0.8,0,0], # left_knee left_hip
        [0,0.8,0], # right_ankle right_knee
        [0,0.6,0.5], # right_knee right_hip
        [0.7,0.7,0.7], # left_hip right_hip
        [0,0.5,0.8], # left_shoulder left_hip
        [0.5,0.9,0], # right_shoulder right_hip
        [0.8,0.1,0.2], # left_shoulder right_shoulder
        [0,1,0], # left_shoulder left_elbow
        [0,0,1], # right_shoulder right_elbow
        [1,0,0], # left_elbow left_wrist
        [0.2, 0.7, 0], # right_elbow right_wrist
        [0,1, 0 ], # left_eye right_eye
        [0,1,0], # nose left_eye
        [0,1,0], # nose right_eye
        [0,1,0], # left_eye left_ear
        [0,1,0], # right_eye right_ear
        [0.3,0.8,0], # left_ear left_shoulder
        [0,0.3,0.8]  # right_ear right_shoulder

    ])
}

def save_batch_image_with_joints(
    batch_image, batch_joints, batch_joints_vis, file_name, nrow=8, padding=2
):
    """
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    """
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    """
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: [batch_size, num_joints, height, width]
    file_name: saved file name
    """
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8,
    )

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(
                resized_image,
                (int(preds[i][j][0]), int(preds[i][j][1])),
                1,
                [0, 0, 255],
                1,
            )
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(
                masked_image,
                (int(preds[i][j][0]), int(preds[i][j][1])),
                1,
                [0, 0, 255],
                1,
            )

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output, prefix):
    if not config.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta["joints"], meta["joints_vis"], f"{prefix}_gt.jpg"
            # input, meta["joints"][0], meta["joints_vis"], f"{prefix}_gt.jpg"
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta["joints_vis"], f"{prefix}_pred.jpg"
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, f"{prefix}_hm_gt.jpg"
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, f"{prefix}_hm_pred.jpg"
        )


def display_random_images(dataset, n: int = 10, seed: int = None):
    if n > 10:
        n = 10
        print(
            f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    if seed is not None:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)
    fig = plt.figure(figsize=(16, 9))

    n_rows = 2
    n_cols = len(random_samples_idx)
    for row in range(n_rows):
        for col, rand_sample in enumerate(random_samples_idx):
            img_path = dataset[rand_sample][3]["image"]
            img_name = str(img_path).split(".")[0].split("/")[-1]

            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            if row == 0:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = dataset[rand_sample][0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            ax.axis("off")
            title = f"name: {img_name}"
            title = title + f"\nshape: {img.shape}"
            plt.title(title)

    return random_samples_idx


def display_images(dataset, indexes, show_keypoints=False, show_text=False):
    fig = plt.figure(figsize=(16, 9))
    n_cols = len(indexes)

    if show_keypoints:
        n_rows = 3
        for row in range(n_rows):
            for col, rand_sample in enumerate(indexes):
                img_path = dataset[rand_sample][3]["image"]
                img_name = str(img_path).split(".")[0].split("/")[-1]
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                if row == 0:
                    org_img = cv2.imread(img_path)
                    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                    img = org_img
                    ax.imshow(img)
                if row == 1:
                    resize_img = dataset[rand_sample][0].numpy()
                    resize_img = resize_img.transpose(1, 2, 0)
                    resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                    img = resize_img
                    ax.imshow(img)
                if row == 2:
                    resize_img = dataset[rand_sample][0].numpy()
                    resize_img = resize_img.transpose(1, 2, 0)
                    resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                    meta = dataset[rand_sample][3]
                    joints_2d = meta["joints"][:, :2]
                    joints_vis = meta["joints_vis"][:, 1]
                    img = resize_img
                    ax.imshow(img)
                    ax = display_keypoints(1, joints_2d, joints_vis, ax, show_text)

                ax.axis("off")
                title = f"name: {img_name}"
                title = title + f"\nshape: {img.shape}"
                plt.title(title)
    else:
        n_rows = 2
        for row in range(n_rows):
            for col, rand_sample in enumerate(indexes):
                img_path = dataset[rand_sample][3]["image"]
                img_name = str(img_path).split(".")[0].split("/")[-1]

                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                if row == 0:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = dataset[rand_sample][0]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                ax.imshow(img)
                ax.axis("off")
                title = f"name: {img_name}"
                title = title + f"\nshape: {img.shape}"
                plt.title(title)


def display_image(input_image):
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")


def display_annotation():
    pass


def display_keypoints(
    draw_non_visible, keypoints, k_vis, ax, show_text=True, draw_not_visible=True,
):  # draw_non_visible -> option to draw (1) or not (0) keypoints that are in the dataset but not visible in the image (occlusions)
    for j in range(len(coco_info["K_NAMES"])):
        n = coco_info["K_NAMES"][j]
        x = keypoints[j, 0]
        y = keypoints[j, 1]

        if k_vis[j] > 0 and (draw_non_visible or (draw_non_visible == 0 and k_vis[j] == 2)):
            ax.scatter(x, y, 30)
            if show_text:
                ax.text(
                    x + 5,
                    y + 5,
                    n,
                    fontsize=10,
                    bbox=dict(facecolor="r", alpha=0.4),
                    color="w",
                )

        if k_vis[j] == 0 and draw_not_visible:
            ax.scatter(x, y, 30)
            if show_text:
                ax.text(
                    x + 5,
                    y + 5,
                    n,
                    fontsize=10,
                    bbox=dict(facecolor="g", alpha=0.4),
                    color="w",
                )
    return ax



def display_keypoints_with_uncertainty(
    draw_non_visible, keypoints, k_vis, uncertainty, ax, show_text=True, bound = 1.0
):
    """

    Args:
        k_vis: MS COCO의 visibility flag
            0: not labeled
            1: labeled but not visible
            2: labeled and visible

    Returns:
        ax
    """
    uncertainty = uncertainty.detach().cpu().numpy()
    uncertainty = np.round(uncertainty, 3)
    for j in range(len(coco_info["K_NAMES"])): # joints의 개수만큼
        n = coco_info["K_NAMES"][j] # 각 joint 이름
        x = keypoints[j, 0] # 각 keypoint의 x값
        y = keypoints[j, 1] # 각 keypoint의 y값

        u = uncertainty[j] # 각 keypoint의 uncertainty값 (x, y)
        ax.scatter(x, y, 30) # 각 joint의 위치를 점으로 나타냄. (크기: 30)
        if show_text:
            # x 또는 y의 값이 bound보다 높으면, 즉 일정 uncertainty보다 높으면, 즉 확실하지 않다고 판단되면 파란색으로 표시
            if u[0] > bound or u[1] > bound:
                ax.text(
                    x + 5,
                    y + 5,
                    f"{n}_{u}",
                    fontsize=10,
                    bbox=dict(facecolor="b", alpha=0.5),
                    color="w",
                )
            # certain하다고 판단되지만 coco에서 label되지 않은 keypoint이면 초록색으로 표시
            elif k_vis[j] == 0:
                ax.text(
                    x + 5,
                    y + 5,
                    f"{n}_{u}",
                    fontsize=10,
                    bbox=dict(facecolor="g", alpha=0.5),
                    color="w",
                )
            # label도 되어있고, visible하다고 되어있지만 uncertainty가 높은 경우 파란색으로 표시
            elif k_vis[j] > 1 and (u[0] > bound or u[1] > bound):
                ax.text(
                    x + 5,
                    y + 5,
                    f"{n}_{u}",
                    fontsize=10,
                    bbox=dict(facecolor="b", alpha=0.5),
                    color="w",
                )
            # 그 외의 경우는 빨간색으로 표시
            else:
                ax.text(
                    x + 5,
                    y + 5,
                    f"{n}_{u}",
                    fontsize=10,
                    bbox=dict(facecolor="r", alpha=0.5),
                    color="w",
                )

    return ax


def display_heatamp(
    ax,
    heatmap,
    heatamp_weight,
    joint_num=17,
    image_shape=(192, 256),
    threshold=0.1,
    alpha=0.2,
    return_keypoint_name=False,
):
    keypoints_name = []
    for i in range(joint_num):
        if heatamp_weight[0][i] == 1:
            keypoints_name.append(coco_info["K_NAMES"][i])
            predicted_heatmap = heatmap[i]
            predicted_heatmap_resized = cv2.resize(
                predicted_heatmap, image_shape, interpolation=cv2.INTER_LINEAR
            )
            predicted_heatmap_resized[predicted_heatmap_resized < threshold] = np.nan
            ax.imshow(predicted_heatmap_resized, cmap="jet", alpha=alpha)

    if return_keypoint_name:
        return ax, keypoints_name
    return ax

def plot_img_with_kp_unc(img, target_keypoints, preds, keys, target_keypoints_weight, uncertainty ):
    fig = plt.figure(1,figsize=(20,16))

    #########################
    ax1 = fig.add_subplot(131)
    image = img[0].permute(1,2,0).detach().cpu().numpy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predicted_keypoints = keys.squeeze(axis=0)
    GT_keypoints_weight = target_keypoints_weight.detach().cpu().numpy()[0]
    ax1 = display_keypoints(1, preds[0], GT_keypoints_weight, ax1)
    ax1.imshow(image)
    plt.title("Predicted Keypoints", fontsize=20)
    ax1.axis("off")

    #########################
    ax1 = fig.add_subplot(132)
    image = img[0].permute(1,2,0).detach().cpu().numpy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predicted_keypoints = keys.squeeze(axis=0)
    ax1 = display_keypoints_with_uncertainty(1, predicted_keypoints.detach().cpu().numpy()*4, target_keypoints_weight.squeeze().detach().cpu().numpy(), uncertainty[0].squeeze(),ax1, bound=1.5)
    # ax1 = display_keypoints_with_uncertainty(1, preds_unc.squeeze(axis=0), GT_keypoints_weight, uncertainty[0].squeeze(),ax1, bound=1.5)
    ax1.imshow(image)
    plt.title("Expected Keypoints", fontsize=20)
    ax1.axis("off")

    #########################
    ax1 = fig.add_subplot(133)
    image = img[0].permute(1,2,0).detach().cpu().numpy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    GT_keypoints = target_keypoints.detach().cpu().numpy()[0]
    ax1 = display_keypoints(1, GT_keypoints, target_keypoints_weight.squeeze().detach().cpu().numpy(), ax1, draw_not_visible=False)
    ax1.imshow(image)
    plt.title("Target Keypoints", fontsize=20)
    ax1.axis("off")
    plt.show()