import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Pool
import ctypes

# copy on write data
g_guidance = None
g_padded_img = None
g_padded_guidance = None
g_output_shape = None
g_gaussion = None
g_exp_lookup = None
g_pad_w = None

# share data
g_output = None
g_weight = None

def g_run(info):
    r = info[0]
    r_num = info[1]
    BYTES = ctypes.sizeof(ctypes.c_double)
    global g_output
    global g_output_shape
    output = np.frombuffer(g_output.get_obj(), offset=g_output_shape[2] * BYTES * r * g_output_shape[1], count=g_output_shape[2] * g_output_shape[1] * r_num).reshape((r_num, g_output_shape[1], g_output_shape[2]))
    global g_weight
    weight = np.frombuffer(g_weight.get_obj(), offset=g_output_shape[2] * BYTES * r * g_output_shape[1], count=g_output_shape[2] * g_output_shape[1] * r_num).reshape((r_num, g_output_shape[1], g_output_shape[2]))
    global g_padded_img
    global g_padded_guidance
    global g_gaussion
    global g_pad_w
    for i in range(-g_pad_w, g_pad_w + 1):
        for j in range(-g_pad_w, g_pad_w + 1):
            if g_padded_guidance.ndim == 2:
                gr = g_exp_lookup[abs(g_padded_guidance[r + i + g_pad_w : r + i + g_pad_w + r_num, j + g_pad_w : j + g_pad_w + g_output_shape[1]] - g_guidance[r : r + r_num, :])]
            else:
                gr = g_exp_lookup[abs(g_padded_guidance[r + i + g_pad_w : r + i + g_pad_w + r_num, j + g_pad_w : j + g_pad_w + g_output_shape[1], 0] - g_guidance[r : r + r_num, :, 0])] * \
                     g_exp_lookup[abs(g_padded_guidance[r + i + g_pad_w : r + i + g_pad_w + r_num, j + g_pad_w : j + g_pad_w + g_output_shape[1], 1] - g_guidance[r : r + r_num, :, 1])] * \
                     g_exp_lookup[abs(g_padded_guidance[r + i + g_pad_w : r + i + g_pad_w + r_num, j + g_pad_w : j + g_pad_w + g_output_shape[1], 2] - g_guidance[r : r + r_num, :, 2])]
            val = gr[:, :, np.newaxis] * g_gaussion[i + g_pad_w, j + g_pad_w]
            output += val * g_padded_img[r + i + g_pad_w : r + i + g_pad_w + r_num, j + g_pad_w : j + g_pad_w + g_output_shape[1]]
            weight += val
    output /= weight

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

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        img = img.astype('int32')
        guidance = guidance.astype('int32')

        global g_guidance
        g_guidance = guidance
        global g_output
        g_output = mp.Array(ctypes.c_double, img.size)
        global g_weight
        g_weight = mp.Array(ctypes.c_double, img.size)
        global g_output_shape
        g_output_shape = img.shape
        global g_padded_img
        g_padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        global g_padded_guidance
        g_padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        SIZE = 2000
        work_lst = [[0, 0]]
        for i in range(img.shape[0]):
            if work_lst[-1][1] * img.shape[1] < SIZE:
                work_lst[-1][1] += 1
            else:
                work_lst.append([i, 1])
        
        # g_run([0, img.shape[0]])

        pool = Pool()
        pool.map(g_run, work_lst)

        output = np.frombuffer(g_output.get_obj()).reshape(img.shape)
        return np.clip(output, 0, 255).astype(np.uint8)