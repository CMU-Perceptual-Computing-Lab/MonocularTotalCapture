import tensorflow as tf
import pickle
from data.BaseReader import BaseReader
import os
import numpy as np


class RHDReader(BaseReader):
    def __init__(self, mode='training', objtype=1, shuffle=False, batch_size=1, crop_noise=False):
        assert objtype == 1
        super(RHDReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode in ('training', 'evaluation')
        self.name = 'RHD'

        self.image_root = '/media/posefs0c/Users/donglaix/Experiments/RHD_published_v2/{}/'.format(mode)
        path_to_db = os.path.join(self.image_root, 'anno_{}.pickle'.format(mode))

        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

        human3d = {'K': [], 'R': [], 't': [], 'distCoef': [], 'left_hand': [], 'left_hand_valid': [], 'right_hand': [], 'right_hand_valid': []}
        img_dirs = []
        mask_dirs = []
        for i, data in db_data.items():
            img_dir = os.path.join(self.image_root, 'color', '{:05d}.png'.format(i))

            if data['uv_vis'][:21, 2].all():
                # add the left hand
                img_dirs.append(img_dir)
                human3d['R'].append(np.eye(3, dtype=np.float32))
                human3d['t'].append(np.zeros((3,), dtype=np.float32))
                human3d['distCoef'].append(np.zeros((5,), dtype=np.float32))
                human3d['K'].append(data['K'].astype(np.float32))

                human3d['left_hand'].append(data['xyz'][:21, :].astype(np.float32))
                human3d['right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))

                mask_dir = os.path.join(self.image_root, 'mask_sep', 'left_{:05d}.png'.format(i))
                mask_dirs.append(mask_dir)

            if data['uv_vis'][21:, 2].all():
                # add the right hand
                img_dirs.append(img_dir)
                human3d['R'].append(np.eye(3, dtype=np.float32))
                human3d['t'].append(np.zeros((3,), dtype=np.float32))
                human3d['distCoef'].append(np.zeros((5,), dtype=np.float32))
                human3d['K'].append(data['K'].astype(np.float32))

                human3d['right_hand'].append(data['xyz'][21:, :].astype(np.float32))
                human3d['left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))

                mask_dir = os.path.join(self.image_root, 'mask_sep', 'right_{:05d}.png'.format(i))
                mask_dirs.append(mask_dir)

        human3d['img_dirs'] = img_dirs
        # human3d['mask_dirs'] = mask_dirs
        self.register_tensor(human3d, {})  # pass in an empty dict because no order needs to be changed
        self.num_samples = len(img_dirs)

    def get(self):
        d = super(RHDReader, self).get(imw=320, imh=320)
        return d


if __name__ == '__main__':
    d = RHDReader(mode='training', shuffle=True, objtype=1, crop_noise=True)
    d.rotate_augmentation = True
    d.blur_augmentation = True
    data_dict = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
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
            sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['scoremap2d'],
                      data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_xyz_local'], data_dict['keypoint_uv_origin'], data_dict['image']])
        image_crop, img_dir, hand2d, hand2d_heatmap, PAF, mask_crop, hand3d, hand2d_origin, image_full = [np.squeeze(_) for _ in values]
        image_v = ((image_crop + 0.5) * 255).astype(np.uint8)
        image_full_v = ((image_full + 0.5) * 255).astype(np.uint8)

        hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:21, :]
        hand3d_detected, _ = PAF_to_3D(hand2d_detected, PAF, objtype=1)
        hand3d_detected = hand3d_detected[:21, :]
        hand_valid = np.ones((21,), dtype=bool)

        fig = plt.figure(1)
        ax1 = fig.add_subplot(241)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        for j in range(21):
            plt.text(hand2d[j, 0], hand2d[j, 1], str(j))

        ax2 = fig.add_subplot(242, projection='3d')
        utils.general.plot3d(ax2, hand3d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.axis('equal')

        ax3 = fig.add_subplot(243, projection='3d')
        utils.general.plot3d(ax3, hand3d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        plt.axis('equal')

        xy, z = plot_all_PAF(PAF, 3)
        ax4 = fig.add_subplot(244)
        ax4.imshow(xy)

        ax5 = fig.add_subplot(245)
        ax5.imshow(z)

        ax6 = fig.add_subplot(246)
        ax6.imshow(image_full_v)
        utils.general.plot2d(ax6, hand2d_origin, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

        ax7 = fig.add_subplot(247)
        mask_3c = np.stack([mask_crop] * 3, axis=2)
        ax7.imshow(mask_3c)

        ax8 = fig.add_subplot(248)
        ax8.imshow((mask_3c * image_v).astype(np.uint8))

        plt.show()
