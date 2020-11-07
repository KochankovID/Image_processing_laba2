import cv2
import numpy as np

from skimage.metrics import mean_squared_error


def comparing_images(
    custom_alg: np.ndarray,
    opencv_alg: np.ndarray,
    first_img_name: str = "first image",
    second_img_name: str = "second image",
):
    log_str = f"Mean_squared_error between {first_img_name} and {second_img_name}: "
    print(log_str, mean_squared_error(custom_alg, opencv_alg))
    create_two_windows(custom_alg, opencv_alg, first_img_name, second_img_name)


def create_two_windows(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Left window",
    title2: str = "Right window",
) -> None:
    """Create two opencv windows with images and titels."""

    cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title2, cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow(title1, 100, 0)
    cv2.moveWindow(title2, 705, 0)

    image1 = cv2.resize(image1, dsize=(600, 600))
    image2 = cv2.resize(image2, dsize=(600, 600))

    cv2.imshow(title1, image1)
    cv2.imshow(title2, image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
