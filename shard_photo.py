"""Script to go from image to 'shards'

(just a cropping of image into smaller pieces to try and put back together)
"""


def grid_shard(image, nrow=3, ncol=3, overlap=30):
    """Break an image into separate rect regions
    
    :param image: image to break-up
    :param nrow: number of rows
    :param ncol: number of columns
    :param overlap: How much should each region overlap with each other (in pixels)
    :return: list of images
    """
    h, w = image.shape[:2]
    col_w = w // ncol
    row_h = h // nrow

    shards = []
    for col_i in range(ncol):
        x0 = col_i * col_w - overlap
        x1 = (col_i + 1) * col_w + overlap

        x0 = max([0, x0])
        x1 = min([w, x1])
        for row_i in range(nrow):
            y0 = row_i * row_h - overlap
            y1 = (row_i + 1) * row_h + overlap

            y0 = max([0, y0])
            y1 = min([h, y1])

            shard = image[y0:y1, x0:x1]
            shards.append(shard)

    return shards


if __name__ == '__main__':
    import os
    import argparse
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help='Input image to break into pieces')
    ap.add_argument('-o', '--output', required=True,
                    help='Location to write pieces to')
    ap.add_argument('-r', '--nrow', default=3, type=int,
                    help='Number of rows to break image into')
    ap.add_argument('-c', '--ncol', default=3, type=int,
                    help='Number of columns to break image into')
    ap.add_argument('-v', '--overlap', default=50, type=int,
                    help='How much should each region overlap with each other (in pixels)')
    args = vars(ap.parse_args())

    input_image = cv2.imread(args['input'])
    input_image_basename = os.path.basename(args['input'])

    im_shards = grid_shard(input_image,
                           args['nrow'],
                           args['ncol'],
                           args['overlap'])

    for i, im_shard in enumerate(im_shards):
        file_name = f'piece_{i}_{input_image_basename}'
        file_path = os.path.join(args['output'], file_name)

        cv2.imwrite(file_path, im_shard)
