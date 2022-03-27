import numpy as np
from pr3_utils import *
from transformation import *

if __name__ == '__main__':
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
    # print(len(features)) # 4
    # print(len(features[0])) # 13289 number of features
    # print(len(features[0][0])) # 3026 timestamps

    velocity = np.vstack((linear_velocity, angular_velocity)) # 6xt

    # Initialize
    T = np.eye(4)
    pose = np.zeros((4, 4, features.shape[2])) # 4x4x3026
    pose[:, :, 0] = T
    mu = np.zeros((4, 4, features.shape[2]))
    mu[:, :, 0] = np.eye(4)
    sigma = np.zeros((6, 6, features.shape[2]))
    sigma[:, :, 0] = np.eye(6)
    noise = 0.001 * np.eye(6)
    for time in range(t.shape[1] - 1):
        tao = t[:, time + 1] - t[:, time]
        se3 = vector2hat(velocity[:, time])
        pose[:, :, time + 1] = pose[:, :, time] @ expm(tao * se3)
        mu[:, :, time + 1] = pose[:, :, time + 1]
        temp = expm(-tao * vector2adjoint(velocity[:, time]))
        sigma[:, :, time + 1] = temp @ sigma[:, :, time] @ temp.T + noise
    fig, ax = visualize_trajectory_2d(pose, "IMU_localization_EKF_prediction", True)
    fig.savefig("IMU_localization_EKF_prediction.jpg", dpi = 1000)
