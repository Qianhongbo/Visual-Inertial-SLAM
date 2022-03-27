# Visual-Inertial SLAM

## Summary

Enabling the autonomous vehicle to perceive the environment is the core technology in autonomous driving. In this project, In this project, I used data from stereo camera with feature extractions from images and IMU measuring linear velocity and angular velocity to implement the Visual Inertial Simultaneous Localization an Mapping(VI-SLAM). Following the instruction, I did IMU Localization via EKF Prediction first to get the trajectory. And then I did Landmark Mapping via EKF Update to update the landmarks on the map. Finally, I combined these two steps to implement an IMU update step based on the stereo-camera observation model to obtain a complete visualinertial SLAM algorithm.

## Data

Under the code folder.

## Requirement

numpy== 1.18.5

matplotlib == 3.2.1

transforms3d == 0.3.1

## Visual-Inertial SLAM

To run the code, run the scripts

IMU_localization_EKF_prediction.py

Landmark_Mapping_via_EKF_Update.py

Visual_Inertial_SLAM.py

## Result

![IMU_localization_EKF_prediction](https://tva1.sinaimg.cn/large/e6c9d24egy1h0p0a9ud6xj20u00u03zs.jpg)

![Landmark_Mapping_via_EKF_Update](https://tva1.sinaimg.cn/large/e6c9d24egy1h0p0aesip6j214o0r0dhs.jpg)

![截屏2022-03-16 00.28.12](https://tva1.sinaimg.cn/large/e6c9d24egy1h0p0ajgsoyj20r20puwgg.jpg)