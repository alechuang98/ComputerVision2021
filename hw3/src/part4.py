import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(0x5EED)
np.random.seed(0x5EED)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        org = np.zeros((len(matches), 3))
        con = np.zeros((len(matches), 3))
        for i, match in enumerate(matches):
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            org[i] = np.array([x1, y1, 1])
            con[i] = np.array([x2, y2, 1])
        
        # TODO: 2. apply RANSAC to choose best H
        idx = np.random.randint(0, len(matches), (ITER, 5))
        sample_org = org[idx]
        sample_con = con[idx]
        H = np.array([solve_homography(sample_org[i], sample_con[i]) for i in range(ITER)])
        res = (H @ org.T).swapaxes(0, 1)
        res /= res[2]
        res = np.moveaxis(res, 0, -1)
        dis = np.linalg.norm(res - con, axis=2)
        id_max = np.argmax(np.count_nonzero(dis < DISTANCE_THRESHOLD, axis=1))
        final_dis = dis[id_max]
        final_org = org[np.argwhere(final_dis < DISTANCE_THRESHOLD)].squeeze()
        final_con = con[np.argwhere(final_dis < DISTANCE_THRESHOLD)].squeeze()
        
        H_inv = np.linalg.inv(solve_homography(final_org, final_con))
        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ H_inv
        # TODO: 4. apply warping
        warping(im2, dst, last_best_H, 0, h_max, 0, w_max, 'b', 
                np.array([[0, 0], [w_max, 0], [w_max, h_max], [0, h_max]]),
                0
                )

    return dst


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3

    DISTANCE_THRESHOLD = 2
    ITER = 20000
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)