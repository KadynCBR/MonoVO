import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
from filterpy import kalman
import matplotlib.pyplot as plt
from detectors import FeatureDetector
from matchers import FeatureMatcher


class MonoVisualOdometry:
    def __init__(self, detector: FeatureDetector, matcher: FeatureMatcher, calib_file: str, debug: bool = False):
        self.detector = detector
        self.matcher = matcher
        self.DEBUG = debug
        self.K, self.P = self._load_calib(calib_file)

    def _load_calib(self, filepath):
        """
        load calib file

        Returns
        -------
        K (ndarray): Interinsics params
        P (ndarray): Projection matrix
        """
        with open(filepath, "r") as f:
            params = np.fromstring(f.readline()[3:], dtype=np.float64, sep=" ")
            print(params)
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _form_transf(R, t):
        # Transformation mat from R, t
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, previous_img, current_img):
        keypoints1, descriptors1 = self.detector.DetectAndCompute(previous_img)  # previous
        keypoints2, descriptors2 = self.detector.DetectAndCompute(current_img)  # current
        if self.DEBUG:
            return self.matcher.match_features_debug(
                descriptors1, keypoints1, descriptors2, keypoints2, previous_img, current_img
            )
        return self.matcher.match_features(descriptors1, keypoints1, descriptors2, keypoints2)

    def get_pose(self, q1, q2):
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)  # CONFIRM CAMERA INTRINSICS (K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R, t)

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transforms = [T1, T2, T3, T4]

        # homogenize k? Come back to this
        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]
        np.set_printoptions(suppress=True)

        positives = []
        for P, T in zip(projections, transforms):
            hom_Q1 = cv2.triangulatePoints(
                self.P, P, q1.T, q2.T
            )  # image coords in first image, image qoord in second, project out into 3d space to get homogenious Q1
            hom_Q2 = T @ hom_Q1
            # un homogenize?
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(
                np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) / np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            )
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if max == 2:
            return R1, np.ndarray.flatten(-t)
        elif max == 3:
            return R2, np.ndarray.flatten(-t)
        elif max == 0:
            return R1, np.ndarray.flatten(t)
        elif max == 1:
            return R2, np.ndarray.flatten(t)
