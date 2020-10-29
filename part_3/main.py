import cv2
import time
import os.path as osp

import sys
sys.path.append('..')

from opencv_scripts.windows_manager import create_two_windows


def cv_median_filter(image1) -> None:
    start = time.time()
    new_image = cv2.medianBlur(image1, 3, None)
    end = time.time()
    print('median filter time: ', end - start)
    create_two_windows(image1, new_image, 'original image', 'new image')


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except (IndexError, AssertionError):
        print('path to the image is not valid! The default path was set!')
        image_path = './src/google.jpg'

    image = cv2.imread(image_path)
    cv_median_filter(image)
