import numpy as np

class StereoCamera():
    def __init__(self, K, b):
        self.K = K
        self.b = b
        self.fsu, self.fsv = self.K[0, 0], self.K[1, 1]
        self.cu, self.cv = self.K[0, 2], self.K[1, 2]
        self.Ks = self.compute_Ks_matrix()

    def compute_Ks_matrix(self):
        """
        compute the Ks matrix from intrinsic parameters
        """
        Ks = np.zeros([4,4])
        Ks[:2, :3] = self.K[:2, :]
        Ks[2:, :3] = self.K[:2, :]
        Ks[2, 3] = -self.fsu * self.b
        return Ks

    def pixel2camera(self, pixels):
        """
        transform pixels of stero camera to camera frame
        pixels: pixels in left and right camera
        Return: homogenous coordinates in camera frame
        """
        d = np.abs(pixels[0] - pixels[2])
        z = self.fsu * self.b / d
        x = z * (pixels[0]  - self.cu) / self.fsu
        y = z * (pixels[1] - self.cv) / self.fsv
        return np.vstack([x, y, z, np.ones(pixels.shape[1])])

    def compute_projection_derivative(self, q):
        """
        lecture 13 slide 5
        """
        temp = np.zeros((4, 4))
        temp[0, 0], temp[1, 1], temp[3, 3] = 1, 1, 1
        temp[0, 2], temp[1, 2], temp[3, 2] = -q[0]/q[2], -q[1]/q[2], -q[3]/q[2]
        return temp / q[2]



