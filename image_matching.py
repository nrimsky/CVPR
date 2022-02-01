import numpy as np


def match_key_points(key_points: np.array) -> np.array:
    """
    :param key_points: n x 2 x 2 array (number of key points x 2 points x dim 2)
    :return: Transformation matrix the minimises the squared error of matching the key points

    We want to find matrix [[h_11, h_12, h_13],
                            [h_21, h_22, h_23],
                            [h_31, h_32, 1]]

    Given correspondences of the form [x_a, y_a, 1] -> [x_b, y_b, 1]

    We can therefore get equations of the form
    h_11 * x_b + h_12 * y_b + h_13 - h_31 * x_b * x_a - h_32 * y_b * x_a - x_a = 0
    h_21 * x_b + h_22 * y_b + h_23 - h_31 * x_b * y_a - h_32 * y_b * y_a - y_a = 0

    for each pair a, b

    These equations can be written in form Ah = 0
    [[x_b, y_b, 1, 0, 0, 0, -x_b * x_a, -y_b * x_a, -x_a], [[h_11],
     [0, 0, 0, x_b, y_b, 1, -x_b * y_a, -y_b * y_a, -y_a]   [h_12],
     ...                                                    [h_13],
                                                            [h_21],
                                                            [h_22],
                                                            [h_23],
                                                            [h_31],
                                                            [h_32],
                                                            [1]]


    """
    a_matrix = []
    for pair in key_points:
        a, b = pair
        x_a, y_a = a
        x_b, y_b = b
        a_matrix.append([x_b, y_b, 1, 0, 0, 0, -x_b * x_a, -y_b * x_a, -x_a])
        a_matrix.append([0, 0, 0, x_b, y_b, 1, -x_b * y_a, -y_b * y_a, -y_a])

    u, s, vh = np.linalg.svd(np.array(a_matrix))
    h = np.transpose(vh)[:, -1].flatten()
    h = h / h[-1]

    return h.reshape((3, 3))


if __name__ == "__main__":
    # These should be close to the identity transformation
    print(match_key_points(np.array([[[1, 4], [1, 4]],
                                     [[2, 2], [2, 2]],
                                     [[1, 2], [1, 2]],
                                     [[5, 2], [5, 2]],
                                     [[8, 9], [8, 9]],
                                     [[0, 2], [0, 2]],
                                     [[1, 3], [1, 3]],
                                     [[7, 2], [7, 2]]])))
    print(match_key_points(np.array([[[1, 4], [1, 4]],
                                     [[2, 2], [2, 2]],
                                     [[3, 9], [3, 9]],
                                     [[1, 2], [1, 2]],
                                     [[5, 2], [5, 2]],
                                     [[8, 9], [8, 9]],
                                     [[0, 2], [0, 2]],
                                     [[1, 3], [1, 3]],
                                     [[7, 2], [7, 2]]])))
