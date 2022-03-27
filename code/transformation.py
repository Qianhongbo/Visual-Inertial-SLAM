import numpy as np
from scipy.linalg import expm

def vector2skew(v):
    """
    convert the vector to skew matrix
    :param v: a 3x1 vector
    :return: skew matrix
    """
    skew_matrix = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]], dtype = float)
    return skew_matrix

def vector2hat(v):
    """
    convert the vector to hat matrix
    :param v: a 6x1 vector
    :return: se(3) twist matrix
            [w_hat, v]
            [0,     0]
    """
    theta_hat = vector2skew(v[3:])
    np.array(v[:3]).reshape((3, 1))
    temp = np.hstack((theta_hat, np.array(v[:3]).reshape((-1, 1))))
    return np.vstack((temp, np.zeros((1,4), dtype = float)))

def vector2adjoint(u):
    """
    convert the vector to joint matrix
    :param v: a 6x1 vector
    :return: se(3) twist matrix
            [w_hat, v_hat]
            [0,     w_hat]
    """
    omega, v = u[3:], u[:3]
    omega_hat, v_hat = vector2skew(omega), vector2skew(v)
    res = np.zeros((6, 6))
    res[:3, :3], res[3:, 3:] = omega_hat, omega_hat
    res[:3, 3:] = v_hat
    return res

def world2camera(Ks, imu_T_cam, T_t_plus_1, mu_t):
    """
    lecture 13 - slide 6
    z_t+1 = Ks * pi(O_T_I * T_t ^ -1 m_j)

    transform coordinates from world frame to camera frame
    Ks: Calibration matrix 4x4
    imu_T_cam: transformation from camera to imu 4x4
    T_t_plus_1: imu pose 4x4
    mu_t: homogenous coordinates of landmarks 4xn
    return: coordinates in camera frame
    """
    temp = np.linalg.inv(imu_T_cam) @ np.linalg.inv(T_t_plus_1) @ mu_t
    res = Ks @ (temp / temp[2])
    return res
