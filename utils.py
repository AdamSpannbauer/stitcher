import math
import cv2
import numpy as np
import imutils


def imshow_max_dim(winname, image, max_width=750):
    h, w = image.shape[:2]
    if w > max_width:
        image = imutils.resize(image, width=max_width)

    cv2.imshow(winname, image)


def image_montage(image_list, n_col, image_shape=(300, 300)):
    n_col = min([len(image_list), n_col])
    n_row = math.ceil(len(image_list) / n_col)

    input_montage = imutils.build_montages(image_list, image_shape=image_shape, montage_shape=(n_col, n_row))

    return input_montage[0]


def pad_to_same_dim(im_a, im_b):
    h_a, w_a = im_a.shape[:2]
    h_b, w_b = im_b.shape[:2]

    pad_x_a = max([0, w_b - w_a])
    pad_y_a = max([0, h_b - h_a])
    pad_x_b = max([0, w_a - w_b])
    pad_y_b = max([0, h_a - h_b])

    im_a = cv2.copyMakeBorder(im_a,
                              top=0, bottom=pad_y_a,
                              left=0, right=pad_x_a,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0, 0])
    im_b = cv2.copyMakeBorder(im_b,
                              top=0, bottom=pad_y_b,
                              left=0, right=pad_x_b,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0, 0])

    return im_a, im_b


def pad_for_join(im_a, im_b):
    h_a, w_a = im_a.shape[:2]
    h_b, w_b = im_b.shape[:2]

    pad_x = (w_a + w_b) // 2 + 1
    pad_y = (h_a + h_b) // 2 + 1

    border_size = max([pad_x, pad_y])

    im_a = cv2.copyMakeBorder(im_a,
                              top=border_size, bottom=border_size,
                              left=border_size, right=border_size,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0, 0])

    im_b = cv2.copyMakeBorder(im_b,
                              top=border_size, bottom=border_size,
                              left=border_size, right=border_size,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0, 0])

    im_a, im_b = pad_to_same_dim(im_a, im_b)

    return im_a, im_b


def layer_overlay(im_a, im_b):
    negative_space = np.where(im_a[:, :, 3] == 0)
    im_a[negative_space] = im_b[negative_space]

    return im_a
