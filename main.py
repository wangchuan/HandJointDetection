import pytoolkit.files as fp
import pytoolkit.tensorflow as tfp
import pytoolkit.vis as vis
import numpy as np

import tensorflow as tf
from config import FLAGS, tfconfig
from data_layer import data_layer
from network import ZFNet
from solver import solver_wrapper

def main():
    net = ZFNet(FLAGS)
    train_data = data_layer(FLAGS, type='train')
    valid_data = data_layer(FLAGS, type='valid')
    saver = tf.train.Saver()
    with tf.Session(config=tfconfig) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        sw = solver_wrapper(net, (train_data, valid_data), sess, saver, FLAGS)
        if FLAGS.mode is 'train':
            sw.train()
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('model restored...')
            if FLAGS.mode is 'test':
                print('acc: %3.2f%%' % (sw.evaluate() * 100.0))
            else:
                sw.train()

if __name__ == "__main__":
    main()