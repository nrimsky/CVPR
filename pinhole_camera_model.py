from typing import Tuple


def object_to_image_space(x: float, y: float, z: float, f: float) -> Tuple[float, float]:
    """
    :param x: x coordinate of object
    :param y: y coordinate of object
    :param z: z coordinate of object
    :param f: focal length of camera
    :return: x_i, y_i (image coordinates in image space)
    """
    y_i = (y * f) / (2 + f)
    x_i = (x * f) / (z + f)
    return x_i, y_i


def image_to_object_space(x_i: float, y_i: float, z: float, f: float) -> Tuple[float, float]:
    """
    :param x_i: x coordinate of object in image
    :param y_i: y coordinate of object in image
    :param z: z coordinate of object in object space
    :param f: focal length of camera
    :return: x, y (x and y coordinates of object in object space)
    """
    x = (x_i * (z + f)) / f
    y = (y_i * (z + f)) / f
    return x, y


def homogeneous_to_cartesian(kx: float, ky: float, kz: float, k: float) -> Tuple[float, float, float]:
    """
    :param kx: 1st homogeneous coordinate
    :param ky: 2nd homogeneous coordinate
    :param kz: 3rd homogeneous coordinate
    :param k: 4th homogeneous coordinate
    :return: Cartesian coordinates
    """
    return kx/k, ky/k, kz/k

