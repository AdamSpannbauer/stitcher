import warnings
import cv2
import imutils
import numpy as np
import utils
import kp_utils


def safe_destroy_window(winname):
    try:
        cv2.destroyWindow(winname)
    except cv2.error:
        pass


def join_ims(im_a, im_b, homography):
    im_a, im_b = utils.pad_for_join(im_a, im_b)
    h, w = im_a.shape[:2]

    warped_bgra_a = cv2.warpAffine(im_a,
                                   homography[:2, :],
                                   (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=[0, 0, 0, 0])

    joined_im = utils.layer_overlay(im_b, warped_bgra_a)

    roi_y, roi_x = np.where(joined_im[:, :, 3] == 255)

    x0 = min(roi_x)
    x1 = max(roi_x)
    y0 = min(roi_y)
    y1 = max(roi_y)

    joined_im = joined_im[y0:y1, x0:x1]

    return joined_im


def join_best_match(image, kps, image_list, kp_list, min_kp, display_matches=False):
    best_match_im_b = None
    best_match_n = 0
    best_match_ind = None
    best_homography = None

    joined_im = None

    for i, (im_b, kps_b) in enumerate(zip(image_list, kp_list)):
        kps_a, features_a = kps
        kps_b, features_b = kps_b

        matches, homography, status = kp_utils.match_keypoints(kps_a, kps_b, features_a, features_b)

        if display_matches:
            matches_im = kp_utils.draw_matches(image, im_b, kps_a, kps_b, matches, status)
            cv2.imshow('Matched KPs', matches_im)
            cv2.waitKey(0)

        if len(matches) > best_match_n:
            best_match_im_b = im_b
            best_match_n = len(matches)
            best_match_ind = i
            best_homography = homography

    if best_homography is not None and best_match_n >= min_kp:
        joined_im = join_ims(image, best_match_im_b, best_homography)
    else:
        best_match_ind = None

    if best_match_n < min_kp:
        min_kp_warning = f'Best match ({best_match_n}) did not meet min_kp threshold ({min_kp})'
        warnings.warn(min_kp_warning)

    return best_match_ind, joined_im


# Should prolly be a class or somethin instead of just passing all these args for state
def _im_join_reduce(image_list, display=False, min_kp=100, max_retry=2, retry_count=0, prev_len=None):
    if prev_len is not None:
        if prev_len == len(image_list):
            retry_count += 1

    prev_len = len(image_list)

    if retry_count > max_retry or len(image_list) == 1:
        safe_destroy_window("Progress")
        return image_list

    kp_feat_list = [kp_utils.detect_and_describe(im) for im in image_list]

    im_a = image_list[0]
    kps_a = kp_feat_list[0]

    del image_list[0]
    del kp_feat_list[0]

    best_match_i, joined_im = join_best_match(im_a, kps_a, image_list, kp_feat_list, min_kp=min_kp)

    if best_match_i is None or joined_im is None:
        image_list = image_list + [im_a]
    else:
        del image_list[best_match_i]
        image_list = image_list + [joined_im]

    if display:
        bgr_image_list = [cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) for im in image_list]
        progress_montage = utils.image_montage(bgr_image_list, n_col=4)

        utils.imshow_max_dim('Progress', progress_montage, max_width=400)
        cv2.waitKey(32)

    return _im_join_reduce(image_list, display=display,
                           min_kp=min_kp, max_retry=max_retry,
                           retry_count=retry_count, prev_len=prev_len)


def im_join_reduce(image_list, display=False, min_kp=100):
    bgra_image_list = [cv2.cvtColor(im, cv2.COLOR_BGR2BGRA) for im in image_list]
    reduced_list = _im_join_reduce(bgra_image_list, display=display, min_kp=min_kp)
    bgr_reduced_list = [cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) for im in reduced_list]

    return bgr_reduced_list
