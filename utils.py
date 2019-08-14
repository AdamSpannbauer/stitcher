import cv2
import numpy as np


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix
    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


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
