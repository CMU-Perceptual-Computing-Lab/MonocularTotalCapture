import tensorflow as tf
from data.Base2DReader import Base2DReader
import os
import pickle
import numpy as np
from utils.keypoint_conversion import GAnerated_to_main as order_dict


class GAneratedReader(Base2DReader):
    def __init__(self, mode='training', objtype=1, shuffle=False, batch_size=1, crop_noise=False):
        super(GAneratedReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode == 'training'
        assert objtype == 1
        self.name = 'GAnerated'

        self.image_root = '/media/posefs1b/Users/donglaix/hand_datasets/GANeratedHands_Release/data/'  # GANerated
        self.path_to_db = '/media/posefs1b/Users/donglaix/hand_datasets/GANeratedHands_Release/data/collected_data.pkl'

        human2d = {'left_hand': [], 'right_hand': [], 'left_hand_valid': [], 'right_hand_valid': []}

        with open(self.path_to_db, 'rb') as f:
            db_data = pickle.load(f)
            # load a tuple of 3 elements: list of img dirs, array of 2D joint, array of 3D joint

        img_dirs = [os.path.join(self.image_root, _) for _ in db_data[0]]
        human2d['right_hand'] = np.zeros((len(img_dirs), 21, 2), dtype=np.float32)
        human2d['right_hand_valid'] = np.zeros((len(img_dirs), 21), dtype=bool)
        human2d['right_hand_3d'] = np.zeros((len(img_dirs), 21, 3), dtype=np.float32)
        human2d['left_hand'] = db_data[1].astype(np.float32)
        human2d['left_hand_valid'] = np.ones((len(img_dirs), 21), dtype=bool)
        human2d['left_hand_3d'] = db_data[2].astype(np.float32)
        human2d['img_dirs'] = img_dirs

        self.num_samples = len(img_dirs)
        self.register_tensor(human2d, order_dict)

    def get(self):
        d = super(GAneratedReader, self).get(imw=256, imh=256)
        return d


if __name__ == '__main__':
    d = GAneratedReader()
    d.rotate_augmentation = True
    d.blur_augmentation = True
    data_dict = d.get()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import utils.general
    from utils.vis_heatmap3d import vis_heatmap3d
    from utils.PAF import plot_PAF, PAF_to_3D, plot_all_PAF

    validation_images = []

    for i in range(d.num_samples):
        print('{}/{}'.format(i + 1, d.num_samples))
        values = \
            sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['hand_valid'], data_dict['scoremap2d'],
                      data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_xyz_origin']])
        image_crop, img_dir, hand2d, hand_valid, hand2d_heatmap, PAF, mask_crop, hand3d = [np.squeeze(_) for _ in values]

        image_name = img_dir.item().decode()
        image_v = ((image_crop + 0.5) * 255).astype(np.uint8)

        hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:21, :]
        hand3d_detected, _ = PAF_to_3D(hand2d_detected, PAF, objtype=1)
        hand3d_detected = hand3d_detected[:21, :]

        fig = plt.figure(1)
        ax1 = fig.add_subplot(231)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

        ax2 = fig.add_subplot(232, projection='3d')
        utils.general.plot3d(ax2, hand3d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        max_range = 0.5 * (np.amax(hand3d_detected, axis=0) - np.amin(hand3d_detected, axis=0)).max()
        center = 0.5 * (np.amax(hand3d_detected, axis=0) + np.amin(hand3d_detected, axis=0))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        ax2.set_xlim(center[0] - max_range, center[0] + max_range)
        ax2.set_ylim(center[1] - max_range, center[1] + max_range)
        ax2.set_zlim(center[2] - max_range, center[2] + max_range)

        ax3 = fig.add_subplot(233, projection='3d')
        utils.general.plot3d(ax3, hand3d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        max_range = 0.5 * (np.amax(hand3d, axis=0) - np.amin(hand3d, axis=0)).max()
        center = 0.5 * (np.amax(hand3d, axis=0) + np.amin(hand3d, axis=0))
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        ax3.set_xlim(center[0] - max_range, center[0] + max_range)
        ax3.set_ylim(center[1] - max_range, center[1] + max_range)
        ax3.set_zlim(center[2] - max_range, center[2] + max_range)

        xy, z = plot_all_PAF(PAF, 3)
        ax4 = fig.add_subplot(234)
        ax4.imshow(xy)

        ax5 = fig.add_subplot(235)
        ax5.imshow(z)

        plt.show()
