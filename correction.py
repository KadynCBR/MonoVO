import numpy as np
import matplotlib.pyplot as plt
from filterpy import kalman


def initialize_kalman_filter(initial_ground_truth_pose):
    # KALMAN FILTER INIT
    kf = kalman.KalmanFilter(dim_x=4, dim_z=2)  # 4 dimension, xpos xvel, ypos yvel
    kf.x = np.array([initial_ground_truth_pose[0, 3], 0, initial_ground_truth_pose[2, 3], 0])
    # measurement function
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    DT = 1 / 10.0  # 30fps deltatime
    # state transiton matrix
    kf.F = np.array([[1.0, DT, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, DT], [0.0, 0.0, 0.0, 1.0]])
    # initial state covariance
    kf.P *= 0.2
    # uncertainty in measurements (trust predictions more)
    kf.R = np.array([[100, 0.0], [0.0, 100]])
    # process noise covariance matrix
    # uncertainty in process model trush measurement more
    kf.Q *= 0.1
    return kf


def extract_unicycle_model(T):
    R = T[:3, :3]
    t = T[:3, 3].T

    x = t[0]
    z = t[2]
    # theta = np.arctan2(R[0, 2], R[2, 2])  # Forward is Z+, yaw around Y
    theta = np.arctan2(R[2, 2], R[0, 2])  # works
    return [x, z, theta]


def plot_pose_2d(ax, x, y, theta, scale=1.0, label=None, color="k"):
    dx = np.cos(theta) * scale
    dy = np.sin(theta) * scale
    ax.arrow(
        x, y, dx, dy, head_width=0.1 * scale, head_length=0.15 * scale, fc=color, ec=color, length_includes_head=True
    )
    if label:
        ax.text(x + 0.01, y + 0.01, f"{label}", fontsize=9, ha="center", color=color)


def diff_model_poses(p1, p2, FPS=1 / 10.0):
    dx, dz, dtheta = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    v = np.sqrt(dx**2 + dz**2) / FPS
    omega = dtheta / FPS
    lateral = dx * (-np.sin(p1[2])) + dz * np.cos(p1[2])
    return dx, dz, dtheta, v, omega, lateral


def detect_outlier(diff_pos, max_linear_vel=12, max_angular_vel=0.7, max_lat=0.4):
    """
    Defaults obtained via diffing ground truth poses and then giving some breathing room.
    """
    dx, dz, dtheta, v, omega, lateral = diff_pos
    outlier_detected = False
    outlier_code = []
    if v > max_linear_vel:
        outlier_code.append("S")
        outlier_detected = True
    if abs(omega) > max_angular_vel:
        outlier_code.append("A")
        outlier_detected = True
    if abs(lateral) > max_lat:
        outlier_code.append("L")
        outlier_detected = True
    return outlier_detected, "".join(outlier_code)


def predict_movement(current_uni, v, w, FPS=1 / 10.0):
    nx = current_uni[0] + v * np.cos(current_uni[2]) * FPS
    nz = current_uni[1] + v * np.sin(current_uni[2]) * FPS
    ntheta = current_uni[2] + w * FPS
    return np.array([nx, nz, ntheta])


def unicycle_to_se3(x, z, theta):
    # Rotation about Y (yaw)
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, 0, z]
    return T


def unicycle_to_se3_MOD(x, z, theta, transform_old):
    # Rotation about Y (yaw)
    R = transform_old[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, 0, z]
    return T


def scale_translation(translation, scale):
    t = translation[:3, 3]
    t = t * scale
    translation[:3, 3] = t
    return translation
