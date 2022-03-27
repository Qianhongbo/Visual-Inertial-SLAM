from pr3_utils import *
from transformation import *
from StereoCamera import StereoCamera
import numpy as np
import matplotlib.pyplot as plt

def get_valid_features_index(features, time):
    """
    the invalid features will be [-1, -1, -1, -1]
    delete these elements
    """
    aList = []
    for i in range(features.shape[1]):
        if not np.array_equal(features[:, i, time], [-1, -1, -1, -1]):
            aList.append(i)
    return aList

def seperate_features(valid_index, features_seen):
    """
    seperate features to seen and unseen
    """
    old_features_index = list(set(valid_index).intersection(set(features_seen)))
    new_features_index = list(set(valid_index).difference(set(features_seen)))
    return old_features_index, new_features_index

def compute_Jacobian(sc, new_features_index, features, imu_T_cam, T, m):
    """
    lecture 13 slide 7
    return: Jacobian matrix 4Ntx3M
    """
    H = np.zeros((4 * len(new_features_index), 3 * features.shape[1]))
    for j in range(len(new_features_index)):
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        o_T_i = np.linalg.inv(imu_T_cam)
        T_inverse = np.linalg.inv(T)
        theBlock = sc.Ks @ sc.compute_projection_derivative(o_T_i @ T_inverse @ m[:,j]) @ o_T_i @ T_inverse @ P.T
        H[4 * j : 4 * (j + 1), 3 * new_features_index[j] : 3 * (new_features_index[j] + 1)] = theBlock
    return H

def compute_Kalman_gain(H, sigma, V):
    '''
    Compute Kalman Gain
    H: Jacobian matrix 4Nt x 3M
    sigma: covariance matrix 3Mx3M
    V: noise matrix 4x4
    Return: K_t+1
    '''
    K = sigma @ H.T @ np.linalg.pinv(H @ sigma @ H.T + np.kron(np.eye(H.shape[0]//4), V))
    return K

def visualize_trajectory_and_mapping_2d(pose,mu,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    ax.scatter(mu[0, :], mu[1, :], 1, label="landmarks")

    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []

        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)

        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.axis('equal')
    ax.set_xlim([-1200, 400])
    ax.set_ylim([-800, 400])
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax

if __name__ == '__main__':
    filename = "./data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    # improve the speed, don't use all of the points
    features = features[:,::100,:]
    velocity = np.vstack((linear_velocity, angular_velocity))  # 6xt
    pose = np.zeros((4, 4, features.shape[2]))  # 4x4x3026
    pose[:, :, 0] = np.eye(4)

    # as homogeneous
    mu = np.zeros([features.shape[0], features.shape[1]])
    sigma = np.eye(3 * features.shape[1])
    sc = StereoCamera(K, b)
    Ks = sc.compute_Ks_matrix()
    features_seen = []

    for time in range(t.shape[1] - 1):
        # compute the trajectory(pose)
        tao = t[:, time + 1] - t[:, time]
        se3 = vector2hat(velocity[:, time])
        pose[:, :, time + 1] = pose[:, :, time] @ expm(tao * se3)

        valid_index = get_valid_features_index(features, time)
        old_features_index, new_features_index = seperate_features(valid_index, features_seen)
        features_seen = list(set(features_seen).union(set(valid_index)))
        # set new features
        if len(new_features_index) > 0:
            # transform from the pixel frame to camera frame...
            coord_camera = sc.pixel2camera(features[:, new_features_index, time])
            # transform from the camera frame to the imu frame and to the world frame...
            coord_world = pose[:, :, time] @ imu_T_cam @ coord_camera
            # initialize the mu as the pose in the world frame...
            mu[:, new_features_index] = coord_world
        # update old features
        if len(old_features_index) > 0:
            # compute the Jacobian matrix
            H = compute_Jacobian(sc, old_features_index, features, imu_T_cam, pose[:,:,time], mu[:, old_features_index])
            # compute the Kalman gain using Jacobian matrix
            Kalman_gain = compute_Kalman_gain(H, sigma, 30 * np.eye(4))
            # update the mean of the old features
            # compute the prediction
            observation_prediction = world2camera(Ks, imu_T_cam, pose[:,:,time], mu[:, old_features_index])
            # compute the innovation, need to r
            innovation = (features[:, old_features_index, time] - observation_prediction).reshape((-1, 1), order='F')
            # update the mean
            mu[:3, :] += (Kalman_gain @ innovation).reshape((3, -1), order='F')
            # update the sigma
            I = np.eye(3 * features.shape[1])
            sigma = (I - Kalman_gain @ H) @ sigma

    plt.scatter(mu[0, :], mu[1, :], 1)
    plt.savefig("landmarks.jpg")
    plt.show()
    fig, ax = visualize_trajectory_and_mapping_2d(pose, mu, "Landmark_Mapping_via_EKF_Update", True)
    fig.savefig("Landmark_Mapping_via_EKF_Update.jpg")
