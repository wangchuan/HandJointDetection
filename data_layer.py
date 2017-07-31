import os
import numpy as np
import cv2
import pytoolkit.files as fp
import utils

class data_layer():
    def __init__(self, FLAGS, type):
        self.batch_size = FLAGS.batch_size
        self.roi_size = 224
        if type is 'train' or 'valid':
            self.data_path = os.path.join(FLAGS.data_path)
            self.data_list = self._load_data_list(os.path.join(self.data_path, type + '_data.txt'))
            self.data, self.label = self._load_data(self.data_list)
        else:
            ValueError('type must be train or valid!')
        self.start_idx = 0
        self._epoch = 0
        self._iteration = 0
        self._iter_cur_epoch = 0

    def next_batch(self):
        if self.start_idx == 0:
            self._iter_cur_epoch = 0
        s = self.start_idx
        e = min(len(self.data), s + self.batch_size)
        data_batch = self._load_patch(self.data[s:e], self.data_list[s:e])
        label_batch = self.label[s:e]

        self._iteration += 1
        self._iter_cur_epoch += 1
        if e == len(self.data):
            self.start_idx = 0
            self._epoch += 1
        else:
            self.start_idx = e
        return data_batch, label_batch

    def _load_data_list(self, filename):
        data_list = []
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                temp = line.rstrip().split(' ')
                p, lb = temp[0], int(temp[1])
                p = os.path.join(self.data_path, p)
                data_list.append((p, lb))
        return data_list

    def _load_data(self, filelist):
        items = []
        lbs = []
        for filename, lb in filelist:
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im = cv2.medianBlur(im, ksize=5)
            items.append(im)
            lbs.append(lb)
        lbs = np.array(lbs)
        return items, lbs

    def _load_patch(self, data, datalist):
        batch_patch = []
        for k, im in enumerate(data):
            filename_img = datalist[k][0]
            path, name, ext = fp.fileparts(filename_img)
            filename_pts = os.path.join(path, name + '.pts')
            pts = utils.read_pts(filename_pts)
            rois = utils.get_rois(im, pts, 0.2, 0.15, self.roi_size) # 224x224x25
            batch_patch.append(rois)
        return np.array(batch_patch)

    def _load_patch2(self, data, datalist):
        batch_patch = []
        ratio = 0.8
        for k, im in enumerate(data):
            h, w = im.shape
            rh, rw = int(h * ratio), int(w * ratio)
            offset_x = np.random.randint(0, w - rw)
            offset_y = np.random.randint(0, h - rh)
            sample = im[offset_y:offset_y+rh, offset_x:offset_x+rw]
            sample = cv2.resize(sample, dsize=(self.roi_size, self.roi_size))
            #cv2.imshow('x', sample)
            #cv2.waitKey(0)
            sample = np.expand_dims(sample, axis=2)
            batch_patch.append(sample)
        return np.array(batch_patch)

    def reset(self):
        self._iteration = 0
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def iter_cur_epoch(self):
        return self._iter_cur_epoch