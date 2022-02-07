import numpy as np
from typing import Tuple


def get_3d_location(f: float, b: float, xl: float, yl: float, xr: float, yr: float) -> Tuple[float, float, float]:
    """
    :param f: Focal length
    :param b: Baseline (distance between cameras - must have parallel optical axes)
    :param xl: Left image x coordinate
    :param yl: Left image y coordinate
    :param xr: Right image x coordinate
    :param yr: Right image y coordinate
    :return: Location of 3D point
    """
    z = f * b / (xl - xr)
    x = xl * z / f
    y = yl * z / f
    return x, y, z

