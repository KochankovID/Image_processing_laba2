import numpy as np
import cv2
import time

import sys
import os.path as osp

if __name__ == "__main__":
    print('Hello world!')
    try:
        image_path = sys.argv[1]
   #     assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except IndexError:
        print('path to the image is not valid! The default path was set!')
    print('Hello world!')
    image = cv2.imread(image_path)
    cv2.imshow('Original image', image)
    new_image = cv2.medianBlur(image, 3, None)
    cv2.imshow('new image', new_image)

    cv2.waitKey(30000)
    cv2.destroyAllWindows()
