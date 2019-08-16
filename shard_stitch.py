import os
import argparse
import random
import cv2
import imutils.paths
import imutils
from shard_photo import grid_shard
from im_join_reduce import im_join_reduce
import utils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input',
                # default='images/headshot.jpeg',
                default='image_pieces',
                help='Path to image/dir build with')
ap.add_argument('-m', '--min_kp', default=40, type=int,
                help="Minimum number of keypoint matches to perform join.")
args = vars(ap.parse_args())

if os.path.isdir(args['input']):
    image_paths = imutils.paths.list_images(args['input'])
    im_list = [cv2.imread(p) for p in image_paths]
else:
    image = cv2.imread(args['input'])
    image = imutils.resize(image, width=min([700, image.shape[1]]))

    im_list = grid_shard(image, nrow=4, ncol=4, overlap=75)
    random.shuffle(im_list)

input_montage = utils.image_montage(im_list, n_col=4)
utils.imshow_max_dim("Input Pieces", input_montage, max_width=400)

reduced_im_list = im_join_reduce(im_list, display=True, min_kp=args['min_kp'])

output_montage = utils.image_montage(reduced_im_list, n_col=4)
utils.imshow_max_dim("Output Pieces", output_montage, max_width=750)

cv2.waitKey(0)
