import numpy as np
from tqdm import tqdm
import argparse
from filterpy import kalman
import matplotlib.pyplot as plt
from mono_vo import MonoVisualOdometry
from detectors import OrbDetector, SiftDetector
from correction import initialize_kalman_filter
from matchers import FLANNMatcher, BFMatcher
from utils import load_gt_poses, load_images, calc_RPE, plot_ape_rpe
from correction import (
    scale_translation,
    extract_unicycle_model,
    diff_model_poses,
    detect_outlier,
    predict_movement,
    unicycle_to_se3_MOD,
)


class PathPlotter:
    def __init__(self, ORB_DETECTOR_FEATURES, RANSAC_THRESHOLD, debug=False, casenum="00"):
        self.debug = debug
        self.casenum = casenum
        if not self.debug:
            return
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], "-b", label="Estimated Path")
        (self.kf_line,) = self.ax.plot([], [], "-r", label="Filtered Estimate")
        self.ax.legend()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(
            f"Pathing nfeatures[{ORB_DETECTOR_FEATURES}] Ransac: {RANSAC_THRESHOLD if RANSAC_THRESHOLD > 0 else 'N/A'}"
        )
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, estimated_path, gt_path, kf_path):
        if not self.debug:
            return
        path_np = np.array(estimated_path)
        kf_path_np = np.array(kf_path)
        self.line.set_data(path_np[:, 0], path_np[:, 1])
        self.kf_line.set_data(kf_path_np[:, 0], kf_path_np[:, 1])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.05)

    def print_output(self, transf, cur_pose):
        print(
            f"""
    Extracted transform difference:
    {str(transf)}

    Current Pose:
    {str(cur_pose)}

    Current Pose used X,Y:
    {str(cur_pose[0,3])} {str(cur_pose[2,3])}
    -----------"""
        )

    def finish(self):
        if not self.debug:
            return
        self.fig.savefig(
            f"{self.casenum}_Pathing_nfeatures-{ORB_DETECTOR_FEATURES}_Ransac-{RANSAC_THRESHOLD if RANSAC_THRESHOLD > 0 else 'NA'}.png"
        )
        plt.ioff()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("StereoEx")
    parser.add_argument("imgdir", type=str)
    parser.add_argument("calib", type=str)
    parser.add_argument("process_num", type=int, default=-1)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()
    images = load_images(args.imgdir, args.process_num + 1 if args.process_num != -1 else -1)

    combinations = [(0, 900, 15)]
    for item in combinations:
        print(f"Starting run [{item[0]}]  nfeatures: [{item[1]}]  ransac: [{item[2]}]")
        ORB_DETECTOR_FEATURES = item[1]
        RANSAC_THRESHOLD = item[2]

        RANSAC = True if RANSAC_THRESHOLD > 0 else False
        orb_detector = OrbDetector(ORB_DETECTOR_FEATURES)
        flann_matcher = FLANNMatcher(ransac=RANSAC, ransacThresh=RANSAC_THRESHOLD)
        bf_matcher = BFMatcher()

        mvo = MonoVisualOdometry(orb_detector, flann_matcher, args.calib, debug=args.debug)
        T = np.eye(4, dtype=np.float64)
        kf = initialize_kalman_filter(T)

        pathplot = PathPlotter(ORB_DETECTOR_FEATURES, RANSAC_THRESHOLD, args.debug)

        estimated_path = []
        kf_path = []
        estimated_poses = []
        for i in tqdm(range(args.process_num), desc="Processing Route"):
            if i == 0:
                current_tracked_pose = T
                previous_pose = extract_unicycle_model(T)
                previous_diff = None
            else:
                q1, q2 = mvo.get_matches(images[i - 1], images[i])
                transf = mvo.get_pose(q1, q2)

                # Correction loop
                t_rel = np.linalg.inv(transf)
                t_rel = scale_translation(t_rel, 0.95)

                # make prediction
                if previous_diff is not None:
                    _, _, _, vel_prev, omega_prev, _ = previous_diff
                    predicted_pose = predict_movement(previous_pose, vel_prev, omega_prev)
                else:
                    predicted_pose = np.zeros((3))

                # make observation
                measurement_pose = current_tracked_pose @ t_rel  # need to inv?
                UNI_measurement_pose = extract_unicycle_model(measurement_pose)
                posediff = diff_model_poses(previous_pose, UNI_measurement_pose)
                outlier, code = detect_outlier(posediff, max_linear_vel=10, max_lat=0.4)

                # determine prediction difference
                prediction_error = np.linalg.norm(np.array(UNI_measurement_pose)[:2] - predicted_pose[:2])
                if args.debug:
                    print(f'{i}\terr: {prediction_error:.4f} ({"OUTLIER" if outlier else ""} {code})')

                # determine what to use, measurement or prediction
                if outlier:
                    selected_pose = predicted_pose
                    current_tracked_pose = unicycle_to_se3_MOD(*predicted_pose, current_tracked_pose)
                else:
                    selected_pose = UNI_measurement_pose
                    current_tracked_pose = measurement_pose

                # update previouses for next loop.
                previous_diff = diff_model_poses(previous_pose, selected_pose)
                previous_pose = selected_pose
                estimated_poses.append(current_tracked_pose)
                # pathplot.print_output(transf, current_tracked_pose)

            x = current_tracked_pose[0, 3]
            y = current_tracked_pose[2, 3]
            kf.predict()
            kf.update(np.array((x, y)))
            kf_path.append((kf.x[0], kf.x[2]))
            estimated_path.append((x, y))
            pathplot.update(estimated_path, None, kf_path)
        pathplot.finish()
