import numpy as np
import cv2
import argparse
from HCD import Harris_corner_detector


def main():
    parser = argparse.ArgumentParser(description='main function of Harris corner detector')
    parser.add_argument('--threshold', default=100., type=float, help='threshold value to determine corner')
    parser.add_argument('--image_path', default='./testdata/ex.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ### TODO ###
    test = Harris_corner_detector(args.threshold)
    output = test.post_processing(test.detect_harris_corners(img_gray))
    for [r, c] in output:
        img = cv2.circle(img, (c, r), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("./results/%d_%s" % (args.threshold, args.image_path.split("/")[-1]), img)

if __name__ == '__main__':
    main()