import cv2
import numpy as np

class FeatureDetector:
    def __init__(self):
        pass

    def DetectAndCompute(self, img):
        pass

class OrbDetector(FeatureDetector):
    def __init__(self, nfeatures=3000):
        self.nfeatures=nfeatures
        self.orb = cv2.ORB_create(self.nfeatures)

    def DetectAndCompute(self, img):
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors
    
class SiftDetector(FeatureDetector):
    def __init__(self, nfeatures=3000):
        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create()

    def DetectAndCompute(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return keypoints, descriptors
    