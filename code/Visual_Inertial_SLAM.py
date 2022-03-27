# %%
import numpy as np
from pr3_utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from StereoCamera import StereoCamera
from Landmark_Mapping_via_EKF_Update import *

def compute_Jacobian_pose(sc, imu_T_cam, T, m, new_features_index):
    """
    compute the Jacobian matrix of the pose
    """
    H = np.zeros((4 * len(new_features_index), 6))
    for i in range(len(new_features_index)):
        o_T_i = np.linalg.inv(imu_T_cam)
        T_inverse = np.linalg.inv(T)
        temp = np.zeros((4, 6))
        temp[:3, :3] = np.eye(3)
        temp[:3, 3:] = -vector2skew((T_inverse @ m[:, i])[:3])
        theBlock = -sc.Ks @ sc.compute_projection_derivative(o_T_i @ T_inverse @ m[:, i]) @ o_T_i @ temp
        H[4 * i:4 * i + 4, :] = theBlock
    return H

if __name__ == '__main__':
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
    # improve the speed, don't use all of the points
    features = features[:, ::100, :]
    velocity = np.vstack((linear_velocity, angular_velocity))  # 6xt
    # initialize all
    pose = np.zeros((4, 4, features.shape[2]))
    pose[:, :, 0] = 1 * np.eye(4)
    mu_pose = np.zeros((4, 4, features.shape[2]))
    mu_pose[:, :, 0] = np.eye(4)
    mu_map = np.zeros((features.shape[0], features.shape[1]))
    sigma = np.zeros((3 * features.shape[1] + 6, 3 * features.shape[1] + 6))
    sigma[:6, :6] = np.eye(6)
    sigma[6:, 6:] = 0.01 * np.eye(3 * features.shape[1])
    sc = StereoCamera(K, b)
    Ks = sc.compute_Ks_matrix()
    features_seen = []

    for time in tqdm(range(t.shape[1] - 1)):
        # prediction step
        tao = t[:, time + 1] - t[:, time]
        se3 = vector2hat(velocity[:, time])
        pose[:, :, time + 1] = pose[:, :, time] @ expm(tao * se3)
        temp = expm(-tao * vector2adjoint(velocity[:, time]))
        sigma[:6, :6] = (temp @ sigma[:6, :6] @ temp.T) + (0.0001) * np.eye(6)
        sigma[:6, 6:] = temp @ sigma[:6, 6:]
        sigma[6:, :6] = sigma[6:, :6] @ temp
        # update step
        valid_index = get_valid_features_index(features, time)
        old_features_index, new_features_index = seperate_features(valid_index, features_seen)
        features_seen = list(set(features_seen).union(set(valid_index)))
        if len(new_features_index) > 0:
            mu_pose[:, :, time] = pose[:, :, time + 1]
            # transform from the pixel frame to camera frame...
            coord_camera = sc.pixel2camera(features[:, new_features_index, time])
            # transform from the camera frame to the imu frame and to the world frame...
            coord_world = pose[:, :, time] @ imu_T_cam @ coord_camera
            # initialize the mu as the pose in the world frame...
            mu_map[:, new_features_index] = coord_world

        if len(old_features_index) > 0:
            # calculate the pose Jacobian matrix
            # calculate the map Jacobian matrix
            # combine the above two matrices
            H_p = compute_Jacobian_pose(sc, imu_T_cam, pose[:, :, time], mu_map[:, old_features_index], old_features_index)
            H_m = compute_Jacobian(sc, old_features_index, features, imu_T_cam, pose[:, :, time], mu_map[:, old_features_index])
            H = np.hstack((H_p, H_m))
            # compute the Kalman gain
            Kalman_gain = compute_Kalman_gain(H, sigma, 30 * np.eye(4))
            # compute the innovation
            observation_prediction = world2camera(Ks, imu_T_cam, pose[:, :, time], mu_map[:, old_features_index])
            innovation = (features[:, old_features_index, time] - observation_prediction).reshape((-1, 1), order='F')
            # update the pose mean and the map mean like in part b
            se3 = vector2hat((Kalman_gain @ innovation)[:6])
            mu_pose[:, :, time + 1] = pose[:, :, time] @ expm(se3)
            mu_map[:3, old_features_index] += ((Kalman_gain @ innovation)[6:].reshape((3, -1), order='F'))[:3, old_features_index]
            # update the map sigma
            I = np.eye(3 * features.shape[1] + 6)
            sigma = (I - Kalman_gain @ H) @ sigma
        else:
            mu_pose[:, :, time + 1] = pose[:, :, time + 1]

    visualize_trajectory_and_mapping_2d(mu_pose, mu_map, "Visual_Inertial_SLAM", False)

