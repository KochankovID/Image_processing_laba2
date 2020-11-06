import sys
import os.path as osp

from typing import Optional


def get_image_path(*, default_path: Optional[str]):
    try:
        image_path = sys.argv[1]
        assert osp.isfile(image_path), '{} is not a file!'.format(image_path)
    except (IndexError, AssertionError):
        print('path to the image is not valid! The default path was set!')
        image_path = osp.join(default_path)
    return image_path
