import os.path as osp

import numpy as np
import cv2

import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))

from image_processing_lib.windows_manager import create_two_windows
from image_processing_lib.time_comparing import get_time
from image_processing_lib.cli_image_argument import get_image_path


def get_gauss_noise(image):
    gaussian_noise = np.zeros(
        (image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8
    )
    cv2.randn(gaussian_noise, (128, 128, 128), (20, 20, 20))
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    return cv2.add(image, gaussian_noise)


def test_gauss_noise(img_path: str):
    image = cv2.imread(img_path)
    noisy = get_time(get_gauss_noise, image)
    create_two_windows(image, noisy)
    return noisy


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(osp.dirname(__file__), "src/rechnaya_vidra_foto.jpg")
    )
    test_gauss_noise(image_path)
