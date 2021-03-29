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
        self.gaussion = self.gaussion[:, :, np.newaxis]
        
        self.exp_lookup = np.zeros((256))
        for i in range(256):
            self.exp_lookup[i] = np.exp(- (i ** 2) / (2 * sigma_r ** 2 * 255 ** 2))
        print('[Build] finish!')

    def get_R(self, r, c, padded_guidance):
        if padded_guidance.ndim == 3:
            return self.exp_lookup[abs(padded_guidance[r - self.pad_w : r + self.pad_w + 1, c - self.pad_w : c + self.pad_w + 1, 0] - padded_guidance[r, c, 0])] * \
                   self.exp_lookup[abs(padded_guidance[r - self.pad_w : r + self.pad_w + 1, c - self.pad_w : c + self.pad_w + 1, 1] - padded_guidance[r, c, 1])] * \
                   self.exp_lookup[abs(padded_guidance[r - self.pad_w : r + self.pad_w + 1, c - self.pad_w : c + self.pad_w + 1, 2] - padded_guidance[r, c, 2])]       
        else:
            return self.exp_lookup[abs(padded_guidance[r - self.pad_w : r + self.pad_w + 1, c - self.pad_w : c + self.pad_w + 1] - padded_guidance[r, c])]


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
                gr = np.expand_dims(self.get_R(i, j, padded_guidance), 2)
                # gr = np.expand_dims(self.exp_lookup[abs(padded_guidance[i - self.pad_w : i + self.pad_w + 1, j - self.pad_w : j + self.pad_w + 1] - padded_guidance[i][j])], 2)
                tmp_matrix = self.gaussion * gr
                total = np.sum(tmp_matrix)
                output[i - self.pad_w][j - self.pad_w] = (tmp_matrix * padded_img[i - self.pad_w : i + self.pad_w + 1, j - self.pad_w : j + self.pad_w + 1]).sum(axis=tuple((0, 1))) / total

        return np.clip(output, 0, 255).astype(np.uint8)