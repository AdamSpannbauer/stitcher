import cv2
import numpy as np
import utils
import kp_utils


def join_ims(im_a, im_b, homography):
    bgra_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2BGRA)
    bgra_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2BGRA)

    bgra_a, bgra_b = utils.pad_for_join(bgra_a, bgra_b)
    h, w = bgra_a.shape[:2]
    warped_bgra_a = cv2.warpPerspective(bgra_a,
                                        homography,
                                        (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=[0, 0, 0, 0])

    joined_im = utils.layer_overlay(bgra_b, warped_bgra_a)

    roi_y, roi_x = np.where(joined_im[:, :, 3] == 255)

    x0 = min(roi_x)
    x1 = max(roi_x)
    y0 = min(roi_y)
    y1 = max(roi_y)

    joined_im = joined_im[y0:y1, x0:x1]
    joined_im = cv2.cvtColor(joined_im, cv2.COLOR_BGRA2BGR)

    return joined_im


def join_best_match(image, kps, image_list, kp_list, display_matches=False):
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

    if best_homography is not None:
        joined_im = join_ims(image, best_match_im_b, best_homography)

    return best_match_ind, joined_im


def im_join_reduce(image_list, display=False):
    if len(image_list) == 1:
        return image_list

    kp_feat_list = [kp_utils.detect_and_describe(im) for im in image_list]

    im_a = image_list[0]
    kps_a = kp_feat_list[0]

    del image_list[0]
    del kp_feat_list[0]

    best_match_i, joined_im = join_best_match(im_a, kps_a, image_list, kp_feat_list)

    if best_match_i is None:
        return image_list

    if display:
        cv2.imshow('Join Progress', joined_im)
        cv2.waitKey(0)

    del image_list[best_match_i]

    image_list = [joined_im] + image_list

    return im_join_reduce(image_list, display=display)


if __name__ == '__main__':
    import imutils.paths

    im_paths = list(imutils.paths.list_images('images'))
    im_list = [cv2.imread(p) for p in im_paths]

    montage = imutils.build_montages(im_list, image_shape=(100, 100), montage_shape=(3, 3))
    cv2.imshow('Input Pieces', montage[0])

    reduced_im_list = im_join_reduce(im_list, display=True)
