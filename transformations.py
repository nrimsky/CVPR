import numpy as np


def make_translate_x_y(t_x, t_y) -> np.array:
    """
    :param t_x: Translation amount in x direction
    :param t_y: Translation amount in y direction
    :return: Transformation matrix
    """
    return np.array([[1, 0, t_x],
                     [0, 1, t_y],
                     [0, 0, 1]])


def make_rotate_theta(theta) -> np.array:
    """
    :param theta: Rotation angle
    :return: Transformation matrix
    """
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def make_euclidean_transform(t_x, t_y, theta) -> np.array:
    """
    :param t_x: Translation amount in x direction
    :param t_y: Translation amount in y direction
    :param theta: Rotation angle
    :return: Transformation matrix
    """

    return np.array([[np.cos(theta), np.sin(theta), t_x],
                     [-np.sin(theta), np.cos(theta), t_y],
                     [0, 0, 1]])


def make_scaling_transform(s) -> np.array:
    """
    :param s: Scale factor
    :return: Transformation matrix
    """

    return np.array([[s, 0, 0],
                     [0, s, 0],
                     [0, 0, 1]])


def make_similarity_transform(theta, s, t_x, t_y) -> np.array:
    """
    :param theta: Rotation angle
    :param s: Scale factor
    :param t_x: Translation amount in x direction
    :param t_y: Translation amount in y direction
    :return: Transformation matrix
    """

    return np.array([[s * np.cos(theta), s * np.sin(theta), t_x],
                     [-s * np.sin(theta), s * np.cos(theta), t_y],
                     [0, 0, 1]])


def make_affine_transform(theta_1, theta_2, s_x, s_y, t_x, t_y) -> np.array:
    """
    :param theta_1: Angle of first rotation (before scaling)
    :param theta_2: Angle of second rotation (after scaling)
    :param s_x: Scale factor in x direction
    :param s_y: Scale factor in y direction
    :param t_x: Translation in x
    :param t_y: Translation in y
    :return: Transformation matrix
    """
    r_1 = np.array([[np.cos(theta_1), np.sin(theta_1)],
                    [-np.sin(theta_1), np.cos(theta_1)]])
    r_2 = np.array([[np.cos(theta_2), np.sin(theta_2)],
                    [-np.sin(theta_2), np.cos(theta_2)]])
    s = np.array([[s_x, 0],
                [0, s_y]])
    rsr = r_2 @ s @ r_1
    t = np.array([[t_x], [t_y]])
    i = np.array([0, 0, 1])
    return np.concatenate((np.concatenate((rsr, t), axis=1), i), axis=0)

