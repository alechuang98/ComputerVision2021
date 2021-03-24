import numpy as np
import cv2
import matplotlib.pyplot as plt


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_harris_corners(self, img):
        ### TODO ####
        # Step 1: Smooth the image by Gaussian kernel
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.5)
        img = cv2.GaussianBlur(img, (3, 3), 1.5)
        # Step 2: Calculate Ix, Iy (1st derivative of image along x and y axis)
        # - Function: cv2.filter2D (kernel = [[1.,0.,-1.]] for Ix or [[1.],[0.],[-1.]] for Iy)
        Ix = cv2.filter2D(img, -1, np.array([[1., 0., -1.]]))
        Iy = cv2.filter2D(img, -1, np.array([[1.],[0.],[-1.]]))
        # Step 3: Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        # Step 4: Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.)
        Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1.)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.)
        Syy = cv2.GaussianBlur(Iyy, (3, 3), 1.)
        # Step 5: Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        # Step 6: Compute the response of the detector by det/(trace+1e-12)
        response = det / (trace + 1e-12)
        return response
    
    def post_processing(self, response):
        ### TODO ###
        # Step 1: Thresholding
        ret, response = cv2.threshold(response, self.threshold, self.threshold, cv2.THRESH_TOZERO)
        # Step 2: Find local maximum
        n = response.shape[0]
        m = response.shape[1]
        local_max = []
        for i in range(n):
            for j in range(m):
                tmp = response[i][j]
                response[i][j] = 0
                if np.max(response[max(0, i - 2) : min(i + 3, n), max(0, j - 2) : min(j + 3, m)]) < tmp:
                    local_max.append([i, j])
                response[i][j] = tmp
                
        return local_max