import tensorflow as tf
from utils.ops import NetworkOps
import numpy as np

ops = NetworkOps


class CPM(object):
    # The original CPM: set input image to right hand, BGR channel order (OpenCV), image scale to x / 256.0 - 0.5, output channel number to 22 (the last one for background)

    def __init__(self, crop_size=256, out_chan=21, withPAF=False, PAFdim=2, numPAF=19, numStage=5, input_chan=3):
        self.name = 'CPM'
        self.out_chan = out_chan
        self.crop_size = crop_size
        self.withPAF = withPAF
        self.PAFdim = PAFdim
        self.numPAF = numPAF
        self.numStage = numStage

    def init(self, weight_path, sess):
        with tf.variable_scope("CPM"):
            data_dict = np.load(weight_path, encoding='latin1').item()
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        var = tf.get_variable(param_name)
                        sess.run(var.assign(data))
        print('Finish loading weight from {}'.format(weight_path))

    def init_pickle(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        import pickle
        import os
        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def init_vgg(self, sess, weight_path='./weights/vgg16.npy'):
        print('initialize from ImageNet pretrained VGG')
        with tf.variable_scope("CPM"):
            data_dict = np.load(weight_path, encoding='latin1').item()
            for op_name in data_dict:
                if not op_name.startswith("conv") or op_name == 'conv5_3':
                    continue
                with tf.variable_scope(op_name, reuse=True):
                    assert len(data_dict[op_name]) == 2
                    for data in data_dict[op_name]:
                        try:
                            if data.ndim == 4:
                                var = tf.get_variable('weights')
                            elif data.ndim == 1:
                                var = tf.get_variable('biases')
                            else:
                                raise Exception
                            sess.run(var.assign(data))
                        except Exception:
                            print('Fail to load {}'.format(op_name))
        print('Finish loading weight from {}'.format(weight_path))

    def inference(self, input_image, train=False):
        with tf.variable_scope("CPM"):
            s = input_image.get_shape().as_list()
            assert s[1] == self.crop_size and s[2] == self.crop_size

            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # conv1_1 ~ conv4_4
            x = input_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3, stride=1, out_chan=chan_num, leaky=False, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            PAF = []

            if not self.withPAF:  # openpose hand net
                x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=512, leaky=False, trainable=train)
                x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=512, leaky=False, trainable=train)
                x = ops.conv_relu(x, 'conv5_1', kernel_size=3, stride=1, out_chan=512, leaky=False, trainable=train)
                x = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=512, leaky=False, trainable=train)

                conv_feature = ops.conv_relu(x, 'conv5_3_CPM', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x = ops.conv_relu(conv_feature, 'conv6_1_CPM', kernel_size=1, stride=1, out_chan=512, leaky=False, trainable=train)
                x = ops.conv(x, 'conv6_2_CPM', kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
                scoremaps = [x]

                for stage_id in range(2, 7):
                    x = tf.concat([x, conv_feature], axis=3, name='concat_stage{}'.format(stage_id))
                    for layer_id in range(1, 6):
                        x = ops.conv_relu(x, 'Mconv{}_stage{}'.format(layer_id, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                    x = ops.conv_relu(x, 'Mconv6_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                    x = ops.conv(x, 'Mconv7_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
                    scoremaps.append(x)

            else:  # with PAF (openpose body net)
                x = ops.conv_relu(x, 'conv4_3_CPM', kernel_size=3, stride=1, out_chan=256, leaky=False, trainable=train)
                conv_feature = ops.conv_relu(x, 'conv4_4_CPM', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)

                x1 = ops.conv_relu(conv_feature, 'conv5_1_CPM_L1', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x1 = ops.conv_relu(x1, 'conv5_2_CPM_L1', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x1 = ops.conv_relu(x1, 'conv5_3_CPM_L1', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x1 = ops.conv_relu(x1, 'conv5_4_CPM_L1', kernel_size=1, stride=1, out_chan=512, leaky=False, trainable=train)
                x1 = ops.conv(x1, 'conv5_5_CPM_L1', kernel_size=1, stride=1, out_chan=self.PAFdim * self.numPAF, trainable=train)

                x2 = ops.conv_relu(conv_feature, 'conv5_1_CPM_L2', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x2 = ops.conv_relu(x2, 'conv5_2_CPM_L2', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x2 = ops.conv_relu(x2, 'conv5_3_CPM_L2', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
                x2 = ops.conv_relu(x2, 'conv5_4_CPM_L2', kernel_size=1, stride=1, out_chan=512, leaky=False, trainable=train)
                x2 = ops.conv(x2, 'conv5_5_CPM_L2', kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)

                scoremaps = [x2]
                PAF.append(x1)

                for stage_id in range(2, 2 + self.numStage):
                    x = tf.concat([x1, x2, conv_feature], axis=3, name='concat_stage{}'.format(stage_id))
                    x1 = ops.conv_relu(x, 'Mconv{}_stage{}_L1'.format(1, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                    x2 = ops.conv_relu(x, 'Mconv{}_stage{}_L2'.format(1, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                    for layer_id in range(2, 6):
                        x1 = ops.conv_relu(x1, 'Mconv{}_stage{}_L1'.format(layer_id, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                        x2 = ops.conv_relu(x2, 'Mconv{}_stage{}_L2'.format(layer_id, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                    x1 = ops.conv_relu(x1, 'Mconv6_stage{}_L1'.format(stage_id), kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                    x2 = ops.conv_relu(x2, 'Mconv6_stage{}_L2'.format(stage_id), kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                    x1 = ops.conv(x1, 'Mconv7_stage{}_L1'.format(stage_id), kernel_size=1, stride=1, out_chan=self.PAFdim * self.numPAF, trainable=train)
                    x2 = ops.conv(x2, 'Mconv7_stage{}_L2'.format(stage_id), kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
                    scoremaps.append(x2)
                    PAF.append(x1)

        return scoremaps, conv_feature, PAF
