import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img_gray.dtype)
    guidances = [img_gray]
 
    ### TODO ###
    with open(args.setting_path, "r") as f:
        f.readline()
        lines = f.readlines()
        for line in lines[:-1]:
            w = line.split(',')
            w = list(map(float, w))
            tmp = np.sum(img_rgb * w, axis=2)
            tmp = tmp.round().astype(np.uint8)
            guidances.append(tmp)
        sigma = lines[-1].split(',')
        sigma_s = int(sigma[1])
        sigma_r = float(sigma[3])
        print(sigma_s, sigma_r)
        # for guidance in guidances:
        JBF = Joint_bilateral_filter(sigma_s, sigma_r)
        origin = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.int32)
        for i in range(len(guidances)):
            tmp = JBF.joint_bilateral_filter(img_rgb, guidances[i]).astype(np.int32)
            cv2.imwrite('./results/guidance_%d_%s' % (i, args.image_path.split('/')[-1]), guidances[i])
            cv2.imwrite('./results/bilateral_%d_%s' % (i, args.image_path.split('/')[-1]), np.flip(tmp, axis=2))
            print("[Error %d]: %d" % (i, np.sum(np.abs(tmp - origin))))

if __name__ == '__main__':
    main()