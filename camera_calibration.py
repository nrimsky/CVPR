import numpy as np


def match_key_points(key_points: np.array) -> dict:
    """
    :param key_points: n x 2 x 3 array (number of key points x 2 points i and w x dim 3) (third dim of i is ignored)
    :return: Transformation matrix the minimises the squared error of matching the key points

    We want to find matrix [[m_11, m_12, m_13, m_14],
                            [m_21, m_22, m_23, m_24],
                            [m_31, m_32, m_33, m_34]]

    Given correspondences of the form [x_i, y_i] -> [x_w, y_w, z_w] (image coordinates -> world coordinates)

    We can therefore get equations of the form
    x_i = m_11 * x_w + m_12 * y_w + m_13 * z_w + m_14 / m_31 * x_w + m_32 * y_w + m_33 * z_w + m_34
    y_i = m_21 * x_w + m_22 * y_w + m_23 * z_w + m_24 / m_31 * x_w + m_32 * y_w + m_33 * z_w + m_34

    Which rearranges to
    x_i(m_31 * x_w + m_32 * y_w + m_33 * z_w + m_34) - (m_11 * x_w + m_12 * y_w + m_13 * z_w + m_14) = 0
    y_i(m_31 * x_w + m_32 * y_w + m_33 * z_w + m_34) - (m_21 * x_w + m_22 * y_w + m_23 * z_w + m_24) = 0


    for each pair a, b

    These equations can be written in form Ah = 0
    [[-x_w, -y_w, -z_w, -1, 0, 0, 0, 0, x_i * x_w, x_i * y_w, x_i * z_w, x_i], [[m_11],
     [0, 0, 0, 0, -x_w, -y_w, -z_w, -1, y_i * x_w, y_i * y_w, y_i * z_w, y_i]   [m_12],
     ...                                                                        [m_13],
                                                                                [m_14],
                                                                                [m_21],
                                                                                [m_22],
                                                                                [m_23],
                                                                                [m_24],
                                                                                [m_31],
                                                                                [m_32],
                                                                                [m_33],
                                                                                [m_34]]


    """
    params = {}
    a_matrix = []
    for pair in key_points:
        i, w = pair
        x_i, y_i, _ = i
        x_w, y_w, z_w = w
        a_matrix.append([-x_w, -y_w, -z_w, -1, 0, 0, 0, 0, x_i * x_w, x_i * y_w, x_i * z_w, x_i])
        a_matrix.append([0, 0, 0, 0, -x_w, -y_w, -z_w, -1, y_i * x_w, y_i * y_w, y_i * z_w, y_i])

    u, s, vh = np.linalg.svd(np.array(a_matrix))
    h = np.transpose(vh)[:, -1].flatten()

    c = h.reshape((3, 4))
    params["Camera matrix"] = c

    a = c[:, :-1]
    b = c[:, -1]
    a_1, a_2, a_3 = a
    b = b.flatten()

    # INTRINSIC PARAMETERS

    # Scaling
    rho = 1 / np.linalg.norm(a_3)
    params["Rho"] = rho

    # Principal point
    x_0 = rho ** 2 * np.dot(a_1, a_3)
    y_0 = rho ** 2 * np.dot(a_2, a_3)
    params["Principal point"] = (x_0, y_0)

    # Skew
    _a13 = np.cross(a_1, a_3)
    _a23 = np.cross(a_2, a_3)
    cos_theta = np.matmul(_a13.transpose(), _a23) / np.linalg.norm(_a13) * np.linalg.norm(_a23)
    theta = np.arccos(cos_theta)
    params["Angle of skew"] = cos_theta

    # Focal
    alpha = rho ** 2 * np.linalg.norm(_a13) * np.sin(theta)
    beta = rho ** 2 * np.linalg.norm(_a23) * np.sin(theta)
    params["Alpha"] = alpha
    params["Beta"] = beta

    # EXTRINSIC PARAMETERS

    # Rotation
    r1 = (_a23 / np.linalg.norm(_a23)).flatten()
    r3 = (a_3 / np.linalg.norm(a_3)).flatten()
    r2 = (np.cross(r3, r1)).flatten()
    r = np.stack((r1, r2, r3))
    params["[EXTRINSIC] Rotation matrix"] = r

    # Translation
    k = np.array([[alpha, -alpha * np.arctan(theta), x_0, 0],
                  [0, beta / np.sin(theta), y_0, 0],
                  [0, 0, 1, 0]])
    params["Intrinsic transformation K"] = k
    inv_k = np.linalg.inv(k[:3, :3])
    t = rho * inv_k @ b
    params["[EXTRINSIC] Translation"] = t

    return params


if __name__ == "__main__":
    camera_params = match_key_points(np.array([[[1, 2, 0], [1, 3, 5]],
                                   [[2, 3, 0], [1, 4, 6]],
                                   [[4, 4, 0], [2, 5, 7]],
                                   [[8, 5, 0], [4, 6, 8]],
                                   [[16, 6, 0], [8, 7, 9]],
                                   [[32, 7, 0], [16, 8, 10]],
                                   [[64, 8, 0], [32, 9, 11]],
                                   [[128, 9, 0], [64, 10, 12]]]))

    c = camera_params["Camera matrix"]

    x = np.matmul(c, np.array([4, 6, 8, 1]))
    print("Should map to approx 8, 5")
    print("Maps to", x[0] / x[-1], x[1] / x[-1])

    x = np.matmul(c, np.array([1, 4, 6, 1]))
    print("Should map to approx 2, 3")
    print("Maps to", x[0] / x[-1], x[1] / x[-1])

    x = np.matmul(c, np.array([64, 10, 12, 1]))
    print("Should map to approx 128, 9")
    print("Maps to", x[0] / x[-1], x[1] / x[-1])

    for param, item in camera_params.items():
        print(f"{param}  =  {item}")
        print("__________________")
