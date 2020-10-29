import sys
import os.path as osp

import numpy as np
import cv2

import sys
sys.path.append('..')

from opencv_scripts.windows_manager import create_two_windows


def get_gauss_noise(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    gaussian_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.randn(gaussian_noise, 128, 20)
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    noisy = cv2.add(image, gaussian_noise)

    create_two_windows(image, noisy)
    return noisy


if __name__ == "__main__":
    image_path = sys.argv[1]
    if not osp.isfile(image_path):
        print('path "{}" to the image is not valid! The default path was set!'.format(image_path))
        image_path = './src/rechnaya_vidra_foto.jpg'
    get_gauss_noise(image_path)

