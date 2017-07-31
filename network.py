import pytoolkit.files as fp
import pytoolkit.tensorflow as tl
import tensorflow as tf
import os

class ZFNet():
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.wd = FLAGS.weight_decay
        sz = FLAGS.image_size
        self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, sz, sz, 25), name='input_image')
        self.ph_label = tf.placeholder(tf.float32, shape=(self.batch_size), name='input_label')

        self.logits = self.inference_v4(self.ph_image)


        self.loss = self.compute_loss(self.ph_label, self.logits)
        self.acc = self.compute_acc(self.ph_label, self.logits)

        self.optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def _conv_layer(self, input, out_channels, conv_ksize, conv_stride_size, pool_ksize, pool_stride_size, name, flag):
        with tf.variable_scope(name) as scope:
            lconv = tl.conv2d(input, out_channels, conv_ksize, conv_stride_size, name='conv')
            lrelu = tl.relu(lconv)
            if flag is True:
                lbn = tl.bn_new(lrelu)
                lpool = tl.max_pool2d(lbn, pool_ksize, pool_stride_size, name='pool')
                return lpool
            else:
                return lrelu

    def inference_v1(self, im):
        l0 = im
        l1 = self._conv_layer(l0,  64, (5,5), (1,1), (2,2), (2,2), name='l1', flag=True)
        l2 = self._conv_layer(l1,  64, (5,5), (1,1), (2,2), (2,2), name='l2', flag=True)
        l3 = self._conv_layer(l2, 128, (3,3), (1,1), (2,2), (2,2), name='l3', flag=True)
        l4 = self._conv_layer(l3, 128, (3,3), (1,1), (0,0), (0,0), name='l4', flag=False)
        l5 = self._conv_layer(l4, 256, (3,3), (1,1), (0,0), (0,0), name='l5', flag=False)
        fc0 = tf.reshape(l5, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 64, name='fc1')
        fc2 = tl.fc(fc1, 32, name='fc2')
        fc3 = tl.fc(fc2,  4, name='fc3')
        return fc3

    def inference_v2(self, im):
        l0 = im
        l1 = self._conv_layer(l0,  32, (5,5), (1,1), (2,2), (2,2), name='l1', flag=True)
        l2 = self._conv_layer(l1,  32, (5,5), (1,1), (2,2), (2,2), name='l2', flag=True)
        l3 = self._conv_layer(l2,  64, (3,3), (1,1), (2,2), (2,2), name='l3', flag=True)
        l4 = self._conv_layer(l3,  64, (3,3), (1,1), (0,0), (0,0), name='l4', flag=False)
        l5 = self._conv_layer(l4, 128, (3,3), (1,1), (0,0), (0,0), name='l5', flag=False)
        fc0 = tf.reshape(l5, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 64, name='fc1')
        fc2 = tl.fc(fc1, 32, name='fc2')
        fc3 = tl.fc(fc2,  4, name='fc3')
        return fc3

    def inference_v3(self, im):
        l0 = im
        l1 = self._conv_layer(l0,  32, (5,5), (1,1), (2,2), (2,2), name='l1', flag=True)
        l2 = self._conv_layer(l1,  32, (5,5), (1,1), (2,2), (2,2), name='l2', flag=True)
        l3 = self._conv_layer(l2,  64, (3,3), (1,1), (2,2), (2,2), name='l3', flag=True)
        l4 = self._conv_layer(l3,  64, (3,3), (1,1), (2,2), (2,2), name='l4', flag=True)
        l5 = self._conv_layer(l4,  64, (3,3), (1,1), (2,2), (2,2), name='l5', flag=True)
        fc0 = tf.reshape(l5, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 64, name='fc1')
        fc2 = tl.fc(fc1, 32, name='fc2')
        fc3 = tl.fc(fc2,  4, name='fc3')
        return fc3

    def inference_v4(self, im):
        l0 = im
        l1 = self._conv_layer(l0,  8, (7,7), (2,2), (2,2), (2,2), name='l1', flag=True)
        l2 = self._conv_layer(l1,  8, (5,5), (2,2), (2,2), (2,2), name='l2', flag=True)
        l3 = self._conv_layer(l2,  16, (3,3), (1,1), (2,2), (2,2), name='l3', flag=True)
        l4 = self._conv_layer(l3,  16, (3,3), (1,1), (2,2), (2,2), name='l4', flag=True)
        l5 = self._conv_layer(l4,  32, (3,3), (1,1), (2,2), (2,2), name='l5', flag=True)
        fc0 = tf.reshape(l5, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 16, name='fc1')
        fc2 = tl.fc(fc1,  8, name='fc2')
        fc3 = tl.fc(fc2,  4, name='fc3')
        return fc3

    def compute_loss(self, labels, logits):
        labels = tf.cast(labels, tf.int32)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        xentropy = tf.reduce_mean(xentropy, name='xentropy_mean')

        t_vars = tf.trainable_variables()
        t_vars_conv = [var for var in t_vars if 'conv' in var.name]
        t_vars_fc = [var for var in t_vars if 'fc' in var.name]
        t_vars = t_vars_conv + t_vars_fc
        for var in t_vars:
            tf.add_to_collection('reg_loss', tf.nn.l2_loss(var))
        l2norm = tf.add_n(tf.get_collection('reg_loss'), name='l2norm')

        return xentropy + self.wd * l2norm

    def compute_acc(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        acc = tf.equal(tf.argmax(logits, 1), labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc






