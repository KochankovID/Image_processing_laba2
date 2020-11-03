import numpy as np
import cv2
import time
import os.path as osp

import sys
sys.path.append('..')

from opencv_scripts.windows_manager import create_two_windows


def get_noise(image1):
    gaussian_noise = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]), dtype=np.uint8)
    cv2.randn(gaussian_noise, (128, 128, 128), (20, 20, 20))
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    return cv2.add(image1, gaussian_noise)


def cv_median_filter(image1) -> None:
    noisy = get_noise(image1)
    start = time.time()
    new_image = cv2.medianBlur(noisy, 3, None)
    end = time.time()
    print('median filter time: ', end - start)
    create_two_windows(noisy, new_image, 'original image', 'new image')


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except (IndexError, AssertionError):
        print('path to the image is not valid! The default path was set!')
        image_path = './src/google.jpg'

    image = cv2.imread(image_path)
    cv_median_filter(image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

