import numpy as np
import cv2
import argparse
import time
import os, glob
from computeDisp import computeDisp

def evaluate(disp_input, disp_gt, scale_factor, threshold=1.0):
    
    disp_input = np.uint8(disp_input * scale_factor)
    disp_input = np.int32(disp_input/scale_factor)
    disp_gt = np.int32(disp_gt/scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    res = np.zeros((h, w)).astype(np.uint8)
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1
                    res[y, x] = 255

    return res, float(nr_error)/nr_pixel

def main():
    parser = argparse.ArgumentParser(description='evaluation function of stereo matching')
    parser.add_argument('--dataset_path', default='./testdata/', help='path to testing dataset')
    args = parser.parse_args()

    config = {'Tsukuba': (15, 16),
              'Venus':   (20, 8),
              'Teddy':   (60, 4),
              'Cones':   (60, 4)}
    total = 0
    for key, val in config.items():
        print('Processing image %s ...'%key)
        t0 = time.time()
        img_left = cv2.imread(os.path.join(args.dataset_path, key, 'img_left.png'))
        img_right = cv2.imread(os.path.join(args.dataset_path, key, 'img_right.png'))
        max_disp, scale_factor = config[key]
        labels = computeDisp(img_left, img_right, max_disp)
        print('[Time] %.4f sec' % (time.time()-t0))

        gt_path = glob.glob(os.path.join(args.dataset_path, key, 'disp_gt.*'))[0]
        if os.path.exists(gt_path):
            img_gt = cv2.imread(gt_path, -1)
            _, error = evaluate(labels, img_gt, scale_factor)
            total += error
            print('[Bad Pixel Ratio] %.2f%%' % (error*100))

    print('[Total Error] %.2f%%' % (total*100))

if __name__ == '__main__':
    main()