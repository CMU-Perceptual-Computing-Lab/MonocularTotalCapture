import tensorflow as tf
import math
import numpy as np


class NetworkOps(object):
    """ Operations that are frequently used within networks. """
    neg_slope_of_relu = 0.01

    @classmethod
    def leaky_relu(cls, tensor, name='relu'):
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu * tensor, name=name)
        return out_tensor

    @classmethod
    def relu(cls, tensor, name='relu_pure'):
        out_tensor = tf.maximum(tensor, tf.constant(0.0, dtype=tf.float32), name=name)
        return out_tensor

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, leaky=True, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        if leaky:
            out_tensor = cls.leaky_relu(tensor, name='out')
        else:
            out_tensor = cls.relu(tensor, name='out')
        return out_tensor

    @classmethod
    def conv3d(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, kernel_size, in_size[4], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv3d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[4]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv3d_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, leaky=True, trainable=True):
        tensor = cls.conv3d(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        if leaky:
            out_tensor = cls.leaky_relu(tensor, name='out')
        else:
            out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def max_pool(cls, bottom, name='pool'):
        pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def upconv(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            kernel_shape = [kernel_size, kernel_size, in_size[3], in_size[3]]
            strides = [1, stride, stride, 1]

            # conv
            kernel = cls.get_deconv_filter(kernel_shape, trainable)
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[2]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def get_deconv_filter(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @staticmethod
    def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
            weights_shape = [in_size[1], out_chan]

            # weight matrix
            weights = tf.get_variable('weights', weights_shape, tf.float32,
                                      tf.contrib.layers.xavier_initializer(), trainable=trainable)
            weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

            # bias
            biases = tf.get_variable('biases', [out_chan], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable)
            biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

            out_tensor = tf.matmul(in_tensor, weights) + biases
            return out_tensor

    @classmethod
    def fully_connected_relu(cls, in_tensor, layer_name, out_chan, trainable=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, trainable)
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu * tensor, name='out')
        return out_tensor

    @staticmethod
    def dropout(in_tensor, keep_prob, evaluation):
        """ Dropout: Each neuron is dropped independently. """
        with tf.variable_scope('dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=tensor_shape))
            return out_tensor

    @staticmethod
    def spatial_dropout(in_tensor, keep_prob, evaluation):
        """ Spatial dropout: Not each neuron is dropped independently, but feature map wise. """
        with tf.variable_scope('spatial_dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=[tensor_shape[0], 1, 1, tensor_shape[3]]))
            return out_tensor
