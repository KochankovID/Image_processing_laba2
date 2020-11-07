import sys
import os.path as osp

import cv2

sys.path.append(osp.dirname(osp.dirname(__file__)))

from image_processing_lib.cli_image_argument import get_image_path
from image_processing_lib.time_comparing import get_time

from part_2.main import median_filter
from part_2.main import gaussian_filter
from part_2.main import blur_filter
from part_2.main import comparing_images
from part_2.main import get_clear_image_and_noisy


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(osp.dirname(__file__), "src/cat.jpg")
    )
    _, noisy = get_clear_image_and_noisy(image_path)

    my_median = get_time(median_filter, noisy, 5)
    opencv_median = get_time(cv2.medianBlur, noisy, 5)
    comparing_images(my_median, opencv_median, 'custom median', 'opencv median')

    my_median = get_time(gaussian_filter, noisy, (5, 5))
    opencv_median = get_time(cv2.GaussianBlur, noisy, (5, 5), sigmaX=1, sigmaY=1)
    comparing_images(my_median, opencv_median, 'custom gauss', 'opencv gauss')

    my_median = get_time(blur_filter, noisy, (5, 5))
    opencv_median = get_time(cv2.blur, noisy, (5, 5))
    comparing_images(my_median, opencv_median, 'custom blur', 'opencv blur')
