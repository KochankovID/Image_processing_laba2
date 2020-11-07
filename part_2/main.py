import os.path as osp

import math
import numpy as np
import cv2

from typing import Callable, Tuple

import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))

from image_processing_lib.windows_manager import comparing_images
from image_processing_lib.time_comparing import get_time
from image_processing_lib.cli_image_argument import get_image_path

from part_1.main import get_gauss_noise


def median_filter(image, kernel_size: int = 5):
    radius = kernel_size // 2

    temp = make_repeat_borders(image, radius, radius)

    result = roll(temp, (kernel_size, kernel_size, 3))
    result = np.median(result, axis=(2, 3, 4)).astype(np.uint8)

    return result


def gaussian_filter(image, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1):
    radius_x = kernel_size[0] // 2
    radius_y = kernel_size[1] // 2

    kernel = np.ones((kernel_size[0], kernel_size[1], 3), np.float64)

    for i in range(-radius_x, radius_x + 1):
        for j in range(-radius_y, radius_y + 1):
            kernel[i + radius_x, j + radius_y] = math.exp(
                -(i ** 2 + j ** 2) / (2 * sigma ** 2)
            )

    kernel /= 2 * math.pi * sigma ** 2 - 1

    temp = make_repeat_borders(image, radius_x, radius_y)

    result = roll(temp, (kernel_size[0], kernel_size[1], 3))
    result = np.multiply(result, kernel)
    result = np.sum(result, axis=(2, 3, 4))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def blur_filter(image, kernel_size: Tuple[int, int] = (5, 5)):
    radius_x = kernel_size[0] // 2
    radius_y = kernel_size[1] // 2

    kernel = np.ones((kernel_size[0], kernel_size[1], 3), np.float32)
    norm = kernel_size[0] * kernel_size[1]
    kernel /= norm

    temp = make_repeat_borders(image, radius_x, radius_y)

    result = roll(temp, (kernel_size[0], kernel_size[1], 3))
    result = np.multiply(result, kernel)
    result = np.sum(result, axis=(2, 3, 4))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def roll(image: np.ndarray, kernel_shape: tuple, dx=1, dy=1, dz=1):
    shape = (
        image.shape[:-3]
        + ((image.shape[-3] - kernel_shape[-3]) // dz + 1,)
        + ((image.shape[-2] - kernel_shape[-2]) // dy + 1,)
        + ((image.shape[-1] - kernel_shape[-1]) // dx + 1,)
        + kernel_shape
    )

    strides = (
        image.strides[:-3]
        + (image.strides[-3] * dz,)
        + (image.strides[-2] * dy,)
        + (image.strides[-1] * dx,)
        + image.strides[-3:]
    )

    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)


def get_clear_image_and_noisy(img_path: str):
    img = cv2.imread(image_path)
    noise = get_gauss_noise(img)
    return img, noise


def make_repeat_borders(image, indent_vertical, indent_horisontal):
    temp = image.copy()

    repeats_vertical = np.ones(temp.shape[0], dtype=np.int64)
    repeats_vertical[0] = repeats_vertical[-1] = indent_vertical + 1

    repeats_horizontal = np.ones(temp.shape[1], dtype=np.int64)
    repeats_horizontal[0] = repeats_horizontal[-1] = indent_horisontal + 1

    temp = np.repeat(temp, repeats_vertical, axis=0)
    temp = np.repeat(temp, repeats_horizontal, axis=1)

    return temp


def test_algorithm(func: Callable, img_path: str):
    img, noisy = get_clear_image_and_noisy(img_path)

    my_blur = get_time(func, noisy)

    comparing_images(my_blur, noisy, func.__name__, "noisy image")
    comparing_images(my_blur, img, func.__name__, "source image")


if __name__ == "__main__":
    image_path = get_image_path(
        default_path=osp.join(osp.dirname(__file__), "src/cat.jpg")
    )
    test_algorithm(median_filter, image_path)
    test_algorithm(gaussian_filter, image_path)
    test_algorithm(blur_filter, image_path)
