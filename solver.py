import tensorflow as tf
import numpy as np
import os

import pytoolkit.vis as vis

class solver_wrapper:
    def __init__(self, net, data_layers, sess, saver, FLAGS):
        self.net = net
        self.train_data = data_layers[0]
        self.valid_data = data_layers[1]
        self.display = FLAGS.disp
        self.epoches = FLAGS.epoches
        self.log_dir = FLAGS.log_path
        self.batch_size = FLAGS.batch_size
        self.sess = sess
        self.saver = saver

    def train(self):
        train_data = self.train_data
        prev_epoch = 0
        saver = self.saver

        fh = vis.fig_handle(len=100)
        while train_data.epoch < self.epoches:
            while True:
                data_batch, label_batch = train_data.next_batch()
                if data_batch.shape[0] == self.batch_size:
                    break
            data_batch = data_batch.astype(np.float32) / 127.5 - 1.0
            feed_dict = {
                self.net.ph_image: data_batch,
                self.net.ph_label: label_batch
            }
            _, loss_val, acc_val = self.sess.run([self.net.optim, self.net.loss, self.net.acc], feed_dict=feed_dict)
            if train_data.iteration % self.display == 0:
                print('Epoch[%03d/%03d] % 8d iters, loss: %3.6f, acc: %3.2f' % (prev_epoch, self.epoches,
                    train_data.iter_cur_epoch, loss_val, acc_val))
            if prev_epoch != train_data.epoch:
                acc_val = self.evaluate() * 100.0
                print('[*] Epoch[%03d/%03d] finished, acc: %3.2f%%' % (prev_epoch, self.epoches, acc_val))
                print('------------------------------------------------')
                fh.update(prev_epoch, acc_val)
                saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'), prev_epoch)
            prev_epoch = train_data.epoch
        fh.savefig()

    def evaluate(self):
        valid_data = self.valid_data
        valid_data.reset()
        total_acc_val, count = 0.0, 0
        while valid_data.epoch < 1:
            while True:
                data_batch, label_batch = valid_data.next_batch()
                if data_batch.shape[0] == self.batch_size:
                    break
            data_batch = data_batch.astype(np.float32) / 127.5 - 1.0
            feed_dict = {
                self.net.ph_image: data_batch,
                self.net.ph_label: label_batch
            }
            acc_val = self.sess.run(self.net.acc, feed_dict=feed_dict)
            total_acc_val += acc_val
            count += 1
        total_acc_val /= count
        return total_acc_val

