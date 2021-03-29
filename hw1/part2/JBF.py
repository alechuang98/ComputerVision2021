import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Pool
import ctypes

# copy on write data
g_padded_img = None
g_padded_guidance = None
g_output_shape = None
g_gaussion = None
g_exp_lookup = None
g_pad_w = None

# share data
g_output = None

def g_R(r, c):
    global g_padded_guidance
    if g_padded_guidance.ndim == 2:
        return g_exp_lookup[abs(g_padded_guidance[r - g_pad_w : r + g_pad_w + 1, c - g_pad_w : c + g_pad_w + 1] - g_padded_guidance[r, c])]
    else:
        return g_exp_lookup[abs(g_padded_guidance[r - g_pad_w : r + g_pad_w + 1, c - g_pad_w : c + g_pad_w + 1, 0] - g_padded_guidance[r, c, 0])] * \
               g_exp_lookup[abs(g_padded_guidance[r - g_pad_w : r + g_pad_w + 1, c - g_pad_w : c + g_pad_w + 1, 1] - g_padded_guidance[r, c, 1])] * \
               g_exp_lookup[abs(g_padded_guidance[r - g_pad_w : r + g_pad_w + 1, c - g_pad_w : c + g_pad_w + 1, 2] - g_padded_guidance[r, c, 2])]

def g_run(pos):
    global g_output
    global g_output_shape
    output = np.frombuffer(g_output.get_obj()).reshape(g_output_shape)
    global g_padded_img
    i = pos[0] + g_pad_w
    j = pos[1] + g_pad_w
    gr = g_R(i, j)[:, :, np.newaxis]
    tmp_matrix = g_gaussion * gr
    output[i - g_pad_w][j - g_pad_w] = (tmp_matrix * g_padded_img[i - g_pad_w : i + g_pad_w + 1, j - g_pad_w : j + g_pad_w + 1]).sum(axis=tuple((0, 1))) / np.sum(tmp_matrix)

class Joint_bilateral_filter(object):
    
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        global g_pad_w
        g_pad_w = self.pad_w
        global g_gaussion
        g_gaussion = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                g_gaussion[i][j] = np.exp(- ((i - self.pad_w) ** 2 + (j - self.pad_w) ** 2) / (2 * sigma_s ** 2))
        g_gaussion = g_gaussion[:, :, np.newaxis]
        
        global g_exp_lookup
        g_exp_lookup = np.zeros((256))
        for i in range(256):
            g_exp_lookup[i] = np.exp(- (i ** 2) / (2 * sigma_r ** 2 * 255 ** 2))
        print('[Build] finish!')

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        img = img.astype('int32')
        guidance = guidance.astype('int32')
        global g_output
        g_output = mp.Array(ctypes.c_double, img.size)
        global g_output_shape
        g_output_shape = img.shape
        global g_padded_img
        g_padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        global g_padded_guidance
        g_padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        pool = Pool()
        pool.map(g_run, np.ndindex((img.shape[0], img.shape[1])))

        '''
        for i in range(self.pad_w, self.pad_w + n):
            for j in range(self.pad_w, self.pad_w + m):
                gr = self.get_R(i, j, self.padded_guidance)[:, :, np.newaxis]
                tmp_matrix = self.gaussion * gr
                self.output[i - self.pad_w][j - self.pad_w] = (tmp_matrix * self.padded_img[i - self.pad_w : i + self.pad_w + 1, j - self.pad_w : j + self.pad_w + 1]).sum(axis=tuple((0, 1))) / np.sum(tmp_matrix)
        '''
        output = np.frombuffer(g_output.get_obj()).reshape(g_output_shape)
        return np.clip(output, 0, 255).astype(np.uint8)