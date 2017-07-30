import tensorflow as tf

def hello():
    print(tf.__version__)

def conv2d(x, output_dim, kernel_size=(3,3), stride_size=(1,1), padding='SAME', stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, b)

def max_pool2d(x, kernel_size=(2,2), stride_size=(2,2), padding='SAME', name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride_size[0], stride_size[1], 1], 
        padding='SAME', name=name)

def relu(x):
    return tf.nn.relu(x, name='relu')

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x)

def fc(x, output_dim, stddev=0.02, bias_start=0.0, name='fc'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(x, w) + b

def deconv2d(x, output_shape, kernel_size, stride_size, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride_size[0], stride_size[1], 1])
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(deconv, b)

def bn_new(x, name='batch_norm'):
    with tf.variable_scope(name):
        x_shape = x.get_shape()
        params_shape = x.shape[-1]
        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        mean, variance = tf.nn.moments(x, axis)
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.0001)

def bn(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, 
            updates_collections=None, epsilon=self.epsilon, 
            scale=True, scope=self.name)

