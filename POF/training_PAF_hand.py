import tensorflow as tf
import os
import sys

from nets.CPM import CPM
from data.DomeReader import DomeReader
from data.TsimonDBReader import TsimonDBReader
from data.RHDReader import RHDReader
from data.STBReader import STBReader
from data.MultiDataset import combineMultiDataset
from data.GAneratedReader import GAneratedReader

import utils.general
import utils.PAF
from utils.multigpu import average_gradients
from tensorflow.python.client import device_lib

num_gpu = sum([_.device_type == 'GPU' for _ in device_lib.list_local_devices()])
fine_tune = False
already_trained = 50000
train_para = {'lr': [1e-4, 1e-5],
              'lr_iter': [80000],
              'max_iter': 160000,
              'show_loss_freq': 10,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots/Final_qual_hand_clear_zoom',
              'finetune_dir': 'snapshots/Final_qual_hand_clear',
              'loss_weight_PAF': 1.0,
              }
PATH_TO_SNAPSHOTS = './{}/model-{}'.format(train_para['finetune_dir'], already_trained)  # only used when USE_RETRAINED is true
ignore_PAF_2D = False

with tf.Graph().as_default(), tf.device('/cpu:0'):
    domereader = DomeReader(mode='training', batch_size=5, shuffle=True, objtype=1, crop_noise=True)
    domereader.crop_scale_noise_sigma = 0.4
    domereader.crop_offset_noise_sigma = 0.2
    rhdreader = RHDReader(mode='training', batch_size=2, shuffle=True, objtype=1, crop_noise=True)
    rhdreader.crop_scale_noise_sigma = 0.4
    rhdreader.crop_offset_noise_sigma = 0.2
    tsimonreader = TsimonDBReader(mode='training', batch_size=1, shuffle=True, objtype=1, crop_noise=True)
    tsimonreader.crop_scale_noise_sigma = 0.4
    tsimonreader.crop_offset_noise_sigma = 0.2
    # ganeratedReader = GAneratedReader(mode='training', batch_size=2, shuffle=True, objtype=1, crop_noise=True)
    data = combineMultiDataset([
        domereader.get(),
        rhdreader.get(),
        tsimonreader.get(),
        # ganeratedReader.get(),
    ],
        name_wanted=['image_crop', 'scoremap2d', 'hand_valid', 'PAF', 'PAF_type', 'mask_crop'])
    # data = domereader.get()
    # stbreader = STBReader(mode='training', batch_size=4, shuffle=True, objtype=1, crop_noise=True)
    # data = stbreader.get()

    for k, v in data.items():
        data[k] = tf.split(v, num_gpu, 0)
    if fine_tune:
        global_step = tf.Variable(already_trained + 1, trainable=False, name="global_step")
    else:
        global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_scheduler = utils.general.LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
    lr = lr_scheduler.get_lr(global_step)
    opt = tf.train.AdamOptimizer(lr)

    tower_grads = []
    tower_losses = []
    tower_losses_PAF = []
    tower_losses_2d = []

    with tf.variable_scope(tf.get_variable_scope()):
        for ig in range(num_gpu):
            with tf.device('/gpu:%d' % ig):

                # build network
                net = CPM(out_chan=22, numPAF=20, crop_size=368, withPAF=True, PAFdim=3)
                predicted_scoremaps, _, predicted_PAFs = net.inference(data['image_crop'][ig], train=True)

                # Loss
                assert len(predicted_scoremaps) == 6
                s = data['scoremap2d'][ig].get_shape().as_list()
                valid = tf.concat([data['hand_valid'][ig], tf.ones((s[0], 1), dtype=tf.bool)], axis=1)
                valid = tf.cast(valid, tf.float32)
                mask_scoremap = tf.tile(tf.expand_dims(data['mask_crop'][ig], axis=3), [1, 1, 1, s[3]])
                loss_2d = 0.0
                # multiply mask_scoremap to mask out the invalid areas
                for ip, predicted_scoremap in enumerate(predicted_scoremaps):
                    resized_scoremap = tf.image.resize_images(predicted_scoremap, (s[1], s[2]))
                    mean_over_pixel = tf.reduce_sum(tf.square((resized_scoremap - data['scoremap2d'][ig]) * mask_scoremap), [1, 2]) / (tf.reduce_sum(mask_scoremap, [1, 2]) + 1e-6)
                    loss_2d_ig = tf.reduce_sum(valid * mean_over_pixel) / (tf.reduce_sum(valid) + 1e-6)
                    loss_2d += loss_2d_ig
                loss_2d /= len(predicted_scoremaps)

                assert 'PAF' in data
                loss_PAF = 0.0
                valid_PAF = tf.cast(utils.PAF.getValidPAF(data['hand_valid'][ig], 1, PAFdim=3), tf.float32)
                # multiply mask_PAF to mask out the invalid areas
                s = data['PAF'][ig].get_shape().as_list()
                mask_PAF = tf.tile(tf.expand_dims(data['mask_crop'][ig], axis=3), [1, 1, 1, s[3]])
                mask_PAF = tf.reshape(mask_PAF, [s[0], s[1], s[2], -1, 3])  # detach x, y, z
                if ignore_PAF_2D:
                    mask_PAF2D = mask_PAF * tf.constant([0, 0, 0], dtype=tf.float32)
                else:
                    mask_PAF2D = mask_PAF * tf.constant([1, 1, 0], dtype=tf.float32)  # for the 2D case
                mask_PAF = tf.where(data['PAF_type'][ig], mask_PAF, mask_PAF2D)  # take out corresponding mask by PAF type
                mask_PAF = tf.reshape(mask_PAF, [s[0], s[1], s[2], -1])
                for ip, pred_PAF in enumerate(predicted_PAFs):
                    resized_PAF = tf.image.resize_images(pred_PAF, (s[1], s[2]), method=tf.image.ResizeMethod.BICUBIC)
                    channelWisePAF = tf.reshape(resized_PAF, [s[0], s[1], s[2], -1, 3])
                    PAF_x2y2 = tf.sqrt(tf.reduce_sum(tf.square(channelWisePAF[:, :, :, :, 0:2]), axis=4)) + 1e-6
                    PAF_normed_x = channelWisePAF[:, :, :, :, 0] / PAF_x2y2
                    PAF_normed_y = channelWisePAF[:, :, :, :, 1] / PAF_x2y2
                    PAF_normed_z = tf.zeros(PAF_normed_x.get_shape(), dtype=tf.float32)
                    normed_PAF = tf.stack([PAF_normed_x, PAF_normed_y, PAF_normed_z], axis=4)
                    normed_PAF = tf.reshape(normed_PAF, [s[0], s[1], s[2], -1])
                    normed_PAF = tf.where(tf.logical_and(tf.not_equal(data['PAF'][ig], 0.0), tf.not_equal(resized_PAF, 0.0)),
                                          normed_PAF, tf.zeros((s[0], s[1], s[2], s[3]), dtype=tf.float32))  # use normed_PAF only in pixels where PAF is not zero
                    final_PAF = tf.where(data['PAF_type'][ig], resized_PAF, normed_PAF)
                    # mean_over_pixel = tf.reduce_sum(tf.square((resized_PAF - data['PAF'][ig]) * mask_PAF), [1, 2]) / (tf.reduce_sum(mask_PAF, [1, 2]) + 1e-6)
                    mean_over_pixel = tf.reduce_sum(tf.square((final_PAF - data['PAF'][ig]) * mask_PAF), [1, 2]) / (tf.reduce_sum(mask_PAF, [1, 2]) + 1e-6)
                    loss_PAF_ig = tf.reduce_sum(valid_PAF * mean_over_pixel) / (tf.reduce_sum(valid_PAF) + 1e-6)
                    loss_PAF += loss_PAF_ig
                loss_PAF /= len(predicted_PAFs)

                loss = loss_2d + loss_PAF * train_para['loss_weight_PAF']
                tf.get_variable_scope().reuse_variables()

                tower_losses.append(loss)
                tower_losses_PAF.append(loss_PAF)
                tower_losses_2d.append(loss_2d)
                grad = opt.compute_gradients(loss)
                tower_grads.append(grad)

    total_loss = tf.reduce_mean(tower_losses)
    total_loss_PAF = tf.reduce_mean(tower_losses_PAF)
    total_loss_2d = tf.reduce_mean(tower_losses_2d)
    grads = average_gradients(tower_grads)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    tf.summary.scalar('loss', total_loss)
    tf.summary.scalar('loss_PAF', total_loss_PAF)
    tf.summary.scalar('loss_2d', total_loss_2d)

    # init weights
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_para['snapshot_dir'] + '/train', sess.graph)

    if not fine_tune:
        start_iter = 0
        net.init_vgg(sess)
    else:
        saver.restore(sess, PATH_TO_SNAPSHOTS)
        start_iter = already_trained + 1

    # snapshot dir
    if not os.path.exists(train_para['snapshot_dir']):
        os.mkdir(train_para['snapshot_dir'])
        print('Created snapshot dir:', train_para['snapshot_dir'])

    # Training loop
    print('Starting to train ...')
    for i in range(start_iter, train_para['max_iter']):
        summary, _, loss_v, loss_2d_v, loss_PAF_v = sess.run([merged, apply_gradient_op, total_loss, total_loss_2d, total_loss_PAF])
        train_writer.add_summary(summary, i)

        if (i % train_para['show_loss_freq']) == 0:
            print('Iteration %d\t Loss %.1e, Loss_2d %.1e, Loss_PAF %.1e' % (i, loss_v, loss_2d_v, loss_PAF_v))
            sys.stdout.flush()

        if (i % train_para['snapshot_freq']) == 0:
            saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
            print('Saved a snapshot.')
            sys.stdout.flush()

    print('Training finished. Saving final snapshot.')
    saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
