import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        self.gaussion = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                self.gaussion[i][j] = np.exp(- ((i - self.pad_w) ** 2 + (j - self.pad_w) ** 2) / (2 * sigma_s ** 2))
        self.exp_lookup = {}
        for i in range(256):
            for j in range(i, 256):
                for k in range(j, 256):
                    self.exp_lookup[i ** 2 + j ** 2 + k ** 2] = np.exp(- (i ** 2 + j ** 2 + k ** 2) / (2 * sigma_r ** 2 * 255 ** 2))

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        n = img.shape[0]
        m = img.shape[1]
        img = img.astype('int32')
        guidance = guidance.astype('int32')
        output = np.zeros_like(img)
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        ### TODO ###
        for i in range(self.pad_w, self.pad_w + n):
            for j in range(self.pad_w, self.pad_w + m):
                gr = np.zeros_like(self.gaussion)
                for di in range(self.wndw_size):
                    for dj in range(self.wndw_size):
                        value = np.sum((padded_guidance[i - self.pad_w + di][j - self.pad_w + dj] - padded_guidance[i][j]) ** 2)
                        gr[di][dj] = self.exp_lookup[value]
                # print(gr)
                # input()
                # print(self.gaussion[:, :, np.newaxis].shape, gr[:, :, np.newaxis].shape, padded_img[i - self.pad_w : i + self.pad_w + 1, j - self.pad_w : j + self.pad_w + 1].shape)
                tmp_matrix = self.gaussion * gr
                total = np.sum(tmp_matrix)
                output[i - self.pad_w][j - self.pad_w] = (tmp_matrix[:, :, np.newaxis] * padded_img[i - self.pad_w : i + self.pad_w + 1, j - self.pad_w : j + self.pad_w + 1]).sum(axis=tuple(range(img.ndim - 1))) / total

        return np.clip(output, 0, 255).astype(np.uint8)