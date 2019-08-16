import argparse
import random
import cv2
import imutils
from shard_photo import grid_shard
from im_join_reduce import im_join_reduce

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default='images/headshot.jpeg',
                help='Path to image to break up and try to put back together')
args = vars(ap.parse_args())

image = cv2.imread(args['input'])
image = imutils.resize(image, width=min([700, image.shape[1]]))

im_list = grid_shard(image, nrow=4, ncol=4, overlap=75)
random.shuffle(im_list)

montage = imutils.build_montages(im_list, image_shape=(100, 100), montage_shape=(3, 3))
cv2.imshow('Input Pieces', montage[0])

reduced_im_list = im_join_reduce(im_list, display=True)
