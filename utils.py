import numpy as np
import cv2

def read_pts(filename):
    """
    the pts file is always x ~ y format,
    and the returned numpy is also x ~ y format
    """
    with open(filename) as f:
        lines = f.readlines()
        n = int(lines[1].split(':')[1])
        pts = np.zeros([n, 2], np.float32)
        for k, s in enumerate(lines[3:n + 3]):
            p = s.split(' ')
            pts[k, 0] = float(p[0])
            pts[k, 1] = float(p[1])
        return pts

def save_pts(filename, rst_pts):
    """
    Note that, the input rst_pts is in x ~ y format,
    but the pts file is always x ~ y format
    """
    with open(filename, 'w') as f:
        f.write('version: 1\n')
        f.write('n_points:  %d\n' % rst_pts.shape[0])
        f.write('{\n')
        for k in range(rst_pts.shape[0]):
            f.write('%f %f\n' % (rst_pts[k, 0], rst_pts[k, 1]))
        f.write('}')

def swap_columns(matrix):
    temp = matrix.copy()
    temp[:,0], temp[:,1] = temp[:,1], temp[:,0].copy()
    return temp

def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    X1 = max(x1, x2)
    X2 = min(x1 + w1, x2 + w2)
    Y1 = max(y1, y2)
    Y2 = min(y1 + h1, y2 + h2)
    return [X1, Y1, max(0, X2 - X1), max(0, Y2 - Y1)]

def get_rois(im, pts, ratio1, ratio2, outsize):
    h, w = im.shape[0], im.shape[1]
    roi_size1 = int(w * ratio1)
    roi_size2 = int(w * ratio2)
    pts = pts.astype(np.int32)
    rois = np.zeros([outsize,outsize,pts.shape[0]], np.uint8)
    for k in range(pts.shape[0]):
        cx, cy = pts[k,0], pts[k,1]
        x, y = cx - roi_size1 // 2, cy - roi_size2 // 2
        ox, oy, ow, oh = overlap([x,y,roi_size1,roi_size1], [0,0,w,h])
        roi = np.zeros([roi_size1, roi_size1], np.uint8)
        roi[oy-y:oy-y+oh, ox-x:ox-x+ow] = im[oy:oy+oh, ox:ox+ow]
        offset_x = np.random.randint(0, roi_size1-roi_size2)
        offset_y = np.random.randint(0, roi_size1-roi_size2)
        roi = roi[offset_y:offset_y+roi_size2, offset_x:offset_x+roi_size2]
        roi = cv2.resize(roi, dsize=(outsize,outsize))
        rois[:,:,k] = roi#cv2.equalizeHist(roi)
        #cv2.imshow('x', roi)
        #cv2.waitKey(0)
    return rois