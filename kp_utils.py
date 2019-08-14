import cv2
import numpy as np
import imutils.feature.factories as kp_factory


DETECTOR = kp_factory.FeatureDetector_create('GFTT')
DESCRIPTOR = kp_factory.DescriptorExtractor_create('BRIEF')
MATCHER = kp_factory.DescriptorMatcher_create('BruteForce')


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kps = DETECTOR.detect(gray)
    kps, features = DESCRIPTOR.compute(image, kps)

    kps = np.float32([kp.pt for kp in kps])

    return kps, features


def match_keypoints(kps_a, kps_b, features_a, features_b, ratio=0.75, reproj_thresh=4.0):
    raw_matches = MATCHER.knnMatch(features_a, features_b, 2)

    matches = []
    for m in raw_matches:
        # Lowe's ratio test to include match
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        pts_a = np.float32([kps_a[i] for (_, i) in matches])
        pts_b = np.float32([kps_b[i] for (i, _) in matches])

        h_mat, status = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thresh)

        return matches, h_mat, status


def draw_matches(image_a, image_b, kps_a, kps_b, matches, status):
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    vis = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype="uint8")
    vis[0:h_a, 0:w_a] = image_a
    vis[0:h_b, w_a:] = image_b

    for (b_i, a_i), s in zip(matches, status):
        if s:
            pt_a = (int(kps_a[a_i][0]), int(kps_a[a_i][1]))
            pt_b = (int(kps_b[b_i][0]) + w_a, int(kps_b[b_i][1]))

            cv2.line(vis, pt_a, pt_b, (0, 255, 0), 1)

    return vis
