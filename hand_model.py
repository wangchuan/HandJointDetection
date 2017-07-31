from pathlib import Path

import menpo.io as mio
from menpofit.clm import CLM, GradientDescentCLMFitter
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
import pytoolkit.files as fp

import numpy as np
import cv2
import os

class HandModel():
    def __init__(self, algo):
        self.model_filename = 'model_' + algo + '.pkl'
        self.algo = algo
        self.fitter = None
        self.template_filename = 'template.pts'

    def load_database(self, path_to_images, max_images=None):
        images = []
        for i in mio.import_images(path_to_images, max_images=max_images, verbose=True):
            if i.n_channels == 3:
                i = i.as_greyscale(mode='luminosity')
            images.append(i)
        return images

    def train(self, train_img_path):
        train_imgs = self.load_database(train_img_path)
        train_imgs = self.equalize_hist(train_imgs)
        if self.algo == 'aam':
            trainer = HolisticAAM(train_imgs, group='PTS', verbose=True, diagonal=120, scales=(0.5, 1.0))
            self.fitter = LucasKanadeAAMFitter(trainer, n_shape=[6, 12], n_appearance=0.5)
            mio.export_pickle(self.fitter, self.model_filename)
            print('aam model trained and exported!')
        elif self.algo == 'asm':
            trainer = CLM(train_imgs, group='PTS', verbose=True, diagonal=120, scales=(0.5, 1.0))
            self.fitter = GradientDescentCLMFitter(trainer, n_shape=[6, 12])
            mio.export_pickle(self.fitter, self.model_filename)
            print('asm model trained and exported!')
        else:
            ValueError('algorithm must be aam or asm!')

    def validate(self, valid_img_path):
        assert(os.path.exists(self.template_filename)), 'template.pts not exists!'
        valid_imgs = self.load_database(valid_img_path)
        valid_imgs = self.equalize_hist(valid_imgs)
        if self.fitter is None and not os.path.exists(self.model_filename):
            ValueError('neither model file nor fitter exists, please train first')
        if self.fitter is None:
            self.fitter = mio.import_pickle(self.model_filename)
        self._validate_without_gt(valid_imgs, self.fitter)

    def _swap_columns(self, matrix):
        temp = matrix.copy()
        temp[:,0], temp[:,1] = temp[:,1], temp[:,0].copy()
        return temp

    def _validate_without_gt(self, valid_imgs, fitter, vis=True):
        Pxy_template = self.read_pts(self.template_filename) # x ~ y format
        menpo_initial_pts = mio.import_landmark_file(self.template_filename).lms

        Pyx_template = self._swap_columns(Pxy_template)

        for im in valid_imgs:
            path = im.path._str
            pathdir, name, ext = fp.fileparts(path)

            Pyx_initial_pts = Pyx_template * np.array([im.pixels.shape[1], im.pixels.shape[2]], np.float32)
            Pxy_initial_pts = self._swap_columns(Pyx_initial_pts)
            menpo_initial_pts.points = Pyx_initial_pts

            fr = fitter.fit_from_shape(im, menpo_initial_pts, gt_shape=None)
            menpo_output_pts = fr.final_shape
            Pxy_output_pts = self._swap_columns(menpo_output_pts.points)

            self.save_pts(os.path.join(pathdir, name + '.pts'), Pxy_output_pts)

            if vis is True:
                Pxy_initial_pts = Pxy_initial_pts.astype(np.int32)
                Pxy_output_pts = Pxy_output_pts.astype(np.int32)
                im_vis = cv2.imread(path)
                for k in range(Pxy_output_pts.shape[0]):
                    cv2.circle(im_vis, (Pxy_initial_pts[k, 0], Pxy_initial_pts[k, 1]), 10, (0, 255, 0), 10)  # green
                    cv2.circle(im_vis, (Pxy_output_pts[k, 0], Pxy_output_pts[k, 1]), 10, (0, 0, 255), 10)  # red
                im_vis = cv2.resize(im_vis, dsize=(0, 0), fx=0.3, fy=0.3)
                if not os.path.exists('./vis/'):
                    os.mkdir('./vis/')
                cv2.imwrite(os.path.join('./vis/', name + '.jpg'), im_vis)

    def read_pts(self, filename):
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

    def save_pts(self, filename, rst_pts):
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

    def equalize_hist(self, imgs):
        for im in imgs:
            data = im.pixels
            data = (data * 255.0).astype(np.uint8)
            h, w = data.shape[1], data.shape[2]
            data = np.reshape(data, [h, w, 1])
            data = cv2.equalizeHist(data)
            data = data.astype(np.float64) / 255.0
            data = np.reshape(data, [1, h, w])
            im.pixels = data
        return imgs

    def compute_average_shape(self, dataset_path):
        filelist = fp.dir(dataset_path, '.pts')
        im_filelist = fp.dir(dataset_path, '.jpg')
        if len(filelist) == 0:
            ValueError('dataset_path contains no pts file!')
        n = 0
        with open(filelist[0]) as f:
            lines = f.readlines()
            n = int(lines[1].split(':')[1])

        P = np.zeros([n, 2], np.float32)
        p0 = np.zeros([1, 2], np.float32)
        for k, filename in enumerate(filelist):
            im_filename = im_filelist[k]
            im = cv2.imread(im_filename)
            pts = self.read_pts(filename)
            row = np.array([im.shape[1], im.shape[0]])
            pts = pts / row
            p0 += pts[0]
            pts -= pts[0]
            P += pts
        p0 /= len(filelist)
        P /= len(filelist)
        P += p0
        P[:, 1], P[:, 0] = P[:, 0], P[:, 1].copy()
        self.save_pts(self.template_filename, P)
        print(self.template_filename + ' saved!')

