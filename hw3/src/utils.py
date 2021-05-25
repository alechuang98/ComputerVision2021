import numpy as np
from matplotlib.path import Path

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for i in range(N):
        A.append([u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0] * v[i][0], -u[i][1] * v[i][0], -v[i][0]])
        A.append([0, 0, 0, u[i][0], u[i][1], 1, -u[i][0] * v[i][1], -u[i][1] * v[i][1], -v[i][1]])
    A = np.array(A)

    # TODO: 2.solve H with A
    _, _, vh = np.linalg.svd(A)
    H = np.reshape(vh[-1, :], (3, 3))
    H /= H[2, 2]
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b', v=None):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
        ones = np.ones_like(x)
        p = np.dstack((x, y, ones)).reshape(-1, 3).T
        points = p.T[:, 0:2]
        p = H_inv @ p
        p /= p[2]
        p = p.T[:, 0:2].reshape(ymax - ymin, xmax - xmin, 2)
        p = np.around(p).astype(np.int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        path = Path(v)
        mask = path.contains_points(points, radius=1e-9).reshape(ymax - ymin, xmax - xmin)
        px = np.where(np.logical_and(p[:, :, 0] < src.shape[1], p[:, :, 0] >= 0), True, False)
        py = np.where(np.logical_and(p[:, :, 1] < src.shape[0], p[:, :, 1] >= 0), True, False)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        pts = np.argwhere(mask & px & py)
        # TODO: 6. assign to destination image with proper masking
        dst[pts[:, 0] + ymin, pts[:, 1] + xmin] = src[p[pts[:, 0], pts[:, 1], 1], p[pts[:, 0], pts[:, 1], 0]]
        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
        ones = np.ones_like(x)
        p = np.dstack((x, y, ones)).reshape(-1, 3).T
        p = H @ p
        p /= p[2]
        p = p.T[:, 0:2].reshape(ymax - ymin, xmax - xmin, 2)
        p = np.around(p).astype(np.int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        px = np.where(np.logical_and(p[:, :, 0] < dst.shape[1], p[:, :, 0] >= 0), True, False)
        py = np.where(np.logical_and(p[:, :, 1] < dst.shape[0], p[:, :, 1] >= 0), True, False)
        flt = np.repeat((px & py)[:, :, np.newaxis], 2, axis=2)
        p = np.where(flt, p, -1)

        # TODO: 6. assign to destination image using advanced array indicing
        dst[p[:, :, 1], p[:, :, 0]] = src[ymin : ymax, xmin : xmax]
        pass

    return dst
