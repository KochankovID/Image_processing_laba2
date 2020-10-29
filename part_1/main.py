import sys
import os.path as osp

import numpy as np
import cv2

import sys
sys.path.append('..')

from opencv_scripts.windows_manager import create_two_windows


def get_gauss_noise(image_path: str):
    image = cv2.imread(image_path)

    gaussian_noise = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
    cv2.randn(gaussian_noise, (128, 128, 128), (20, 20, 20))
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    noisy = cv2.add(image, gaussian_noise)

    create_two_windows(image, noisy)
    return noisy


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except:
        print('path to the image is not valid! The default path was set!')
        image_path = './src/rechnaya_vidra_foto.jpg'
    get_gauss_noise(image_path)


