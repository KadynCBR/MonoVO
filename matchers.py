import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self):
        pass

    def match_features(self, descriptors1, keypoints1, descriptors2, keypoints2):
        pass

    def match_features_debug(self, descriptors1, keypoints1, descriptors2, keypoints2, img1, img2):
        pass

    def ransac_filtering(self, keypoints1, keypoints2, matches, ransacReprojThreshold = 40.0):
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        # Estimate affine transform using RANSAC
        affine_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransacReprojThreshold)
        # print(f"Inliers after RANSAC: {np.sum(inliers)} / {len(inliers)}")
        inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]

        return inlier_matches


# algorithm 6 = FLANN_INDEX_LSH
# algorithm 1 = FLANN_INDEX_KDTREE
class FLANNMatcher(FeatureMatcher):
    def __init__(self, k=2, algorithm=6, table_number=6, key_size=12, multi_probe_levels=1, checks=50, ransac=False, ransacThresh=40.0):
        super().__init__()
        self.k = k
        self.algorithm = algorithm
        self.table_number = table_number
        self.key_size = key_size
        self.multi_probe_levels = multi_probe_levels
        self.checks = checks
        self.ransac = ransac
        self.ransacThresh = ransacThresh

        index_params = dict(algorithm=self.algorithm, table_number=self.table_number, key_size=self.key_size, multi_probe_levels=self.multi_probe_levels)
        search_params = dict(checks=50)        
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def _get_good_matches(self, descriptors1, descriptors2):
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=self.k)
        # store good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            # if the distance is close to each other in the image plane, its a good match
            if m.distance < 0.8*n.distance: 
                good.append(m)
        return good

    def match_features(self, descriptors1, keypoints1, descriptors2, keypoints2):
        good = self._get_good_matches(descriptors1, descriptors2)
        if (self.ransac):
            good = self.ransac_filtering(keypoints1, keypoints2, good, self.ransacThresh)
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])
        return q1, q2
    
    def match_features_debug(self, descriptors1, keypoints1, descriptors2, keypoints2, img1, img2):
        good = self._get_good_matches(descriptors1, descriptors2)
        if (self.ransac):
            good = self.ransac_filtering(keypoints1, keypoints2, good, self.ransacThresh)
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])        
        draw_params = dict(matchColor = -1, singlePointColor = None, matchesMask = None, flags=2)
        img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
        cv2.imshow("image", cv2.resize(img3, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(10)
        return q1, q2

    
class BFMatcher(FeatureMatcher):
    def __init__(self, matchtype = cv2.NORM_L2, crossCheck=True, ransac=False):
        self.matchtype = matchtype
        self.crossCheck = crossCheck
        self.ransac = ransac
        self.bfmatcher = cv2.BFMatcher(matchtype, crossCheck)

    def _get_good_matches(self, descriptors1, descriptors2):
        # since we're doing crosscheck we dont need to do lowe ratio test, already filtered
        matches = self.bfmatcher.match(descriptors1, descriptors2)
        return matches
    
    def match_features(self, descriptors1, keypoints1, descriptors2, keypoints2):
        good = self._get_good_matches(descriptors1, descriptors2)
        if (self.ransac):
            good = self.ransac_filtering(keypoints1, keypoints2, good)
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])        
        return q1, q2
    
    def match_features_debug(self, descriptors1, keypoints1, descriptors2, keypoints2, img1, img2):
        good = self._get_good_matches(descriptors1, descriptors2)
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])        
        draw_params = dict(matchColor = -1, singlePointColor = None, matchesMask = None, flags=2)
        img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
        cv2.imshow("image", cv2.resize(img3, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(10)
        return q1, q2