import cv2
import numpy as np
from tqdm import tqdm
import os
from filterpy import kalman
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_images(image_dir, num=-1):
    imagefiles = [os.path.join(image_dir, file) for file in sorted(os.listdir(image_dir))]
    return [cv2.imread(imgfn, cv2.IMREAD_GRAYSCALE) for imgfn in tqdm(imagefiles[:num], desc="Loading images")]


def load_gt_poses(filepath):
    poses = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=" ")
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses


def calc_RPE(gt_path, estimate_path, delta=1):
    translation_errors = []
    for i in range(len(gt_path) - delta):
        T_gt_rel = np.array(gt_path[i + delta]) - np.array(gt_path[i])
        T_est_rel = np.array(estimate_path[i + delta]) - np.array(estimate_path[i])

        T_error = T_est_rel - T_gt_rel
        trans_error = np.linalg.norm(T_error)
        translation_errors.append(trans_error)

    return translation_errors, np.mean(translation_errors)


def plot_ape_rpe(gt_positions, est_positions, delta=1):
    gt_positions = np.array(gt_positions)
    est_positions = np.array(est_positions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], "b-", label="Ground Truth")
    ax.plot(est_positions[:, 0], est_positions[:, 1], "r--", label="Filtered Estimate")
    for gt, est in zip(gt_positions, est_positions):
        ax.plot([gt[0], est[0]], [gt[1], est[1]], "g-", alpha=0.5)

    ax.set_title("APE (green lines)")
    ax.set_aspect("equal")
    ax.legend()
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
