import numpy as np
import cv2.ximgproc as xip
import cv2

def computeCost(l, r, disp):
    # a.shape = (h, w, 3, 8)
    h, w = l.shape[:2]
    cost = np.count_nonzero(l[:, disp:w] == r[:, 0:(w-disp)], axis=-1).astype(np.uint8)
    res = np.pad(cost, pad_width=[(0, 0), (disp, 0), (0, 0)])
    return res

def getClosestNeighbor(row_l, row_r, w, direction):
    res = np.zeros(w)
    pre = -1
    lst = [x for x in range(w)]
    lst = lst[::direction]
    for i in lst:
        if row_l[i] <= i and row_r[i - row_l[i]] == row_l[i]:
            pre = i
            res[i] = row_l[i]
        else:
            if pre == -1:
                res[i] = np.Inf
            else:
                res[i] = row_l[pre]
    return res

def computeDisp(Il, Ir, max_disp, cfg):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    dx = [0, 1, 2, 0, 2, 0, 1, 2]
    dy = [0, 0, 0, 1, 1, 2, 2, 2]
    Il_pad = np.pad(Il, [(1, 1), (1, 1), (0, 0)])
    Ir_pad = np.pad(Ir, [(1, 1), (1, 1), (0, 0)])
    Il_pattern = np.zeros((h, w, ch, len(dx))).astype(np.uint8)
    Ir_pattern = np.zeros((h, w, ch, len(dx))).astype(np.uint8)

    for i in range(len(dx)):
        Il_pattern[:, :, :, i] = Il > Il_pad[dy[i]:dy[i]+h, dx[i]:dx[i]+w]
        Ir_pattern[:, :, :, i] = Ir > Ir_pad[dy[i]:dy[i]+h, dx[i]:dx[i]+w]
    
    Il_map = np.zeros((max_disp+1, h, w, ch)).astype(np.float32)
    Ir_map = np.zeros((max_disp+1, h, w, ch)).astype(np.float32)
    for disp in range(max_disp+1):
        Il_map[disp] = computeCost(Il_pattern, Ir_pattern, disp)
        Ir_map[disp] = np.flip(computeCost(np.flip(Ir_pattern, axis=1), np.flip(Il_pattern, axis=1), disp), axis=1)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for i in range(max_disp+1):
        Il_map[i] = xip.jointBilateralFilter(Il, Il_map[i], d=-1, sigmaColor=cfg['sigmaColor'], sigmaSpace=cfg['sigmaSpace'])
        Ir_map[i] = xip.jointBilateralFilter(Ir, Ir_map[i], d=-1, sigmaColor=cfg['sigmaColor'], sigmaSpace=cfg['sigmaSpace'])
    Il_map = np.sum(Il_map, axis=-1)
    Ir_map = np.sum(Ir_map, axis=-1)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    Il_map = np.argmax(Il_map, axis=0)
    Ir_map = np.argmax(Ir_map, axis=0)
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for i in range(h):
        ln = getClosestNeighbor(Il_map[i], Ir_map[i], w, 1)
        rn = getClosestNeighbor(Il_map[i], Ir_map[i], w, -1)
        Il_map[i] = np.minimum(ln, rn).astype(np.float32)
    
    Il_g = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    Il_map = Il_map.astype(np.uint8)
    labels = xip.weightedMedianFilter(Il_g, Il_map, r=cfg['r'])

    return labels.astype(np.uint8)
    