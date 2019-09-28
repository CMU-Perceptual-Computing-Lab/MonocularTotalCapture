import tensorflow as tf
import pickle
import os
from utils.ops import NetworkOps as ops


class handSegNet:
    def __init__(self):
        pass

    def init_sess(self, sess):
        file_name = './weights/handsegnet-rhd.pickle'
        exclude_var_list = []
        assert os.path.exists(file_name), "File not found."
        with open(file_name, 'rb') as fi:
            weight_dict = pickle.load(fi)
            weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
            if len(weight_dict) > 0:
                init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                sess.run(init_op, init_feed)
                print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def inference_detection(self, image, train=False):
        """ HandSegNet: Detects the hand in the input image by segmenting it.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 2] tf.float32 tensor, Scores for the hand segmentation classes
        """
        with tf.variable_scope('HandSegNet'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 4]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv5_1', kernel_size=3, stride=1, out_chan=512, trainable=train)
            encoding = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv6_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv6_2', kernel_size=1, stride=1, out_chan=2, trainable=train)
            scoremap_list.append(scoremap)

            # upsample to full size
            s = image.get_shape().as_list()
            scoremap_list_large = [tf.image.resize_images(x, (s[1], s[2])) for x in scoremap_list]

        return scoremap_list_large
