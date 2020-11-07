import cv2
import os.path as osp

import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))

from image_processing_lib.windows_manager import create_two_windows
from image_processing_lib.time_comparing import get_time
from image_processing_lib.cli_image_argument import get_image_path

from part_1.main import get_gauss_noise


def cv_median_filter(image1) -> None:
    noisy = get_gauss_noise(image1)
    new_image = get_time(cv2.medianBlur, noisy, 3, None)
    create_two_windows(noisy, new_image, "original image", "new image")


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(osp.dirname(__file__), "src/google.jpg")
    )
    image = cv2.imread(image_path)
    cv_median_filter(image)
