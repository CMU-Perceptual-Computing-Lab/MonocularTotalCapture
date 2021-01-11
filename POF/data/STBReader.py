import tensorflow as tf
import os
import numpy as np
from data.BaseReader import BaseReader
import pickle
from data.collect_stb import PATH_TO_DATASET, K, Rl, Rr, tl, tr, TRAIN_SEQS, TEST_SEQS
from utils.keypoint_conversion import STB_to_main


class STBReader(BaseReader):
    def __init__(self, mode='training', objtype=1, shuffle=False, batch_size=1, crop_noise=False):
        assert objtype == 1
        super(STBReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode in ('training', 'evaluation')
        self.name = 'STB'

        self.image_root = PATH_TO_DATASET
        path_to_db = './data/stb_collected.pkl'
        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

        if mode == 'training':
            mode_data = db_data[0]
            SEQS = TRAIN_SEQS
        else:
            mode_data = db_data[1]
            SEQS = TEST_SEQS
        assert mode_data.shape[0] == len(SEQS) * 1500

        hand3d = np.tile(mode_data, [2, 1, 1]).astype(np.float32)
        hand3d[:, 0] = 2 * hand3d[:, 0] - hand3d[:, 9]
        self.num_samples = hand3d.shape[0]
        Ks = np.array([K] * self.num_samples, dtype=np.float32)
        Rs = np.array([Rl] * mode_data.shape[0] + [Rr] * mode_data.shape[0], dtype=np.float32)
        ts = np.array([tl] * mode_data.shape[0] + [tr] * mode_data.shape[0], dtype=np.float32)
        distCoef = np.zeros([self.num_samples, 5], dtype=np.float32)
        left_hand_valid = np.ones([self.num_samples, 21], dtype=bool)
        img_dirs = [os.path.join(self.image_root, seq, 'BB_left_{}.png').format(i) for seq in SEQS for i in range(1500)] + \
                   [os.path.join(self.image_root, seq, 'BB_right_{}.png'.format(i)) for seq in SEQS for i in range(1500)]

        human3d = {'K': Ks, 'R': Rs, 't': ts, 'distCoef': distCoef,
                   'left_hand': hand3d, 'left_hand_valid': left_hand_valid, 'img_dirs': img_dirs,
                   'right_hand': np.zeros([self.num_samples, 21, 3], dtype=np.float32), 'right_hand_valid': np.zeros([self.num_samples, 21], dtype=bool)}
        self.register_tensor(human3d, STB_to_main)

    def get(self):
        d = super(STBReader, self).get(imw=640, imh=480)
        return d


if __name__ == '__main__':
    d = STBReader(mode='evaluation', shuffle=True, objtype=1, crop_noise=True)
    data_dict = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import utils.general
    from utils.PAF import plot_PAF, PAF_to_3D, plot_all_PAF

    for i in range(d.num_samples):
        print('{}/{}'.format(i + 1, d.num_samples))
        values = \
            sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['left_hand_valid'], data_dict['scoremap2d'],
                      data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_xyz_local'], data_dict['keypoint_uv_origin'], data_dict['image']])
        image_crop, img_dir, hand2d, hand_valid, hand2d_heatmap, PAF, mask_crop, hand3d, hand2d_origin, image_full = [np.squeeze(_) for _ in values]
        image_v = ((image_crop + 0.5) * 255).astype(np.uint8)
        image_full_v = ((image_full + 0.5) * 255).astype(np.uint8)

        hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:21, :]
        hand3d_detected, _ = PAF_to_3D(hand2d_detected, PAF, objtype=1)
        hand3d_detected = hand3d_detected[:21, :]

        fig = plt.figure(1)
        ax1 = fig.add_subplot(231)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        for j in range(21):
            plt.text(hand2d[j, 0], hand2d[j, 1], str(j))

        ax2 = fig.add_subplot(232, projection='3d')
        utils.general.plot3d(ax2, hand3d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.axis('equal')

        ax3 = fig.add_subplot(233, projection='3d')
        utils.general.plot3d(ax3, hand3d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        plt.axis('equal')

        xy, z = plot_all_PAF(PAF, 3)
        ax4 = fig.add_subplot(234)
        ax4.imshow(xy)

        ax5 = fig.add_subplot(235)
        ax5.imshow(z)

        ax6 = fig.add_subplot(236)
        ax6.imshow(image_full_v)
        utils.general.plot2d(ax6, hand2d_origin, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

        plt.show()
