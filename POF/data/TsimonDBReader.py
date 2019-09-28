import tensorflow as tf
import numpy as np
import json
from data.Base2DReader import Base2DReader
import os
from utils.keypoint_conversion import tsimon_to_main as order_dict


class TsimonDBReader(Base2DReader):
    def __init__(self, mode='training', objtype=1, shuffle=False, batch_size=1, crop_noise=False):
        super(TsimonDBReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode == 'training'
        assert objtype == 1
        self.name = 'Tsimon'

        self.image_root = '/media/posefs0c/Users/donglaix/tsimon/'
        self.path_to_db = ['/media/posefs0c/Users/donglaix/tsimon/hands_v12.json', '/media/posefs0c/Users/donglaix/tsimon/hands_v13.json', '/media/posefs0c/Users/donglaix/tsimon/hands_v143.json']

        human2d = {'left_hand': [], 'right_hand': [], 'left_hand_valid': [], 'right_hand_valid': []}
        img_dirs = []

        for filename in self.path_to_db:
            with open(filename) as f:
                filedata = json.load(f)
            for ihand, hand_data in enumerate(filedata['root']):
                joint2d = np.array(hand_data['joint_self'])
                human2d['right_hand'].append(joint2d[:, :2].astype(np.float32))
                human2d['right_hand_valid'].append(joint2d[:, 2].astype(bool))
                human2d['left_hand'].append(np.zeros((21, 2), dtype=np.float32))
                human2d['left_hand_valid'].append(np.zeros((21,), dtype=bool))

                img_dir = os.path.join(self.image_root, '/'.join(hand_data['img_paths'].split('/')[5:]))
                img_dirs.append(img_dir)

        human2d['img_dirs'] = img_dirs
        self.num_samples = len(img_dirs)
        self.register_tensor(human2d, order_dict)

    def get(self):
        d = super(TsimonDBReader, self).get(imw=1920, imh=1080)
        return d


if __name__ == '__main__':
    dataset = TsimonDBReader(mode='training', shuffle=True, objtype=1, crop_noise=True)
    data_dict = dataset.get()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import utils.general
    from utils.PAF import plot_PAF, PAF_to_3D, plot_all_PAF

    for i in range(dataset.num_samples):
        print('{}/{}'.format(i + 1, dataset.num_samples))
        values = \
            sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['scoremap2d'],
                      data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_uv_origin'], data_dict['image'],
                      data_dict['left_hand_valid'], data_dict['right_hand_valid']])
        image_crop, img_dir, hand2d, hand2d_heatmap, PAF, mask_crop, hand2d_origin, image_full, left_hand_valid, right_hand_valid \
            = [np.squeeze(_) for _ in values]
        image_v = ((image_crop + 0.5) * 255).astype(np.uint8)
        image_full_v = ((image_full + 0.5) * 255).astype(np.uint8)

        hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:21, :]
        hand_valid = right_hand_valid

        fig = plt.figure(1)
        ax1 = fig.add_subplot(241)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        for j in range(21):
            plt.text(hand2d[j, 0], hand2d[j, 1], str(j))

        xy, z = plot_all_PAF(PAF, 3)
        ax4 = fig.add_subplot(244)
        ax4.imshow(xy)

        ax6 = fig.add_subplot(246)
        ax6.imshow(image_full_v)
        utils.general.plot2d(ax6, hand2d_origin, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

        ax7 = fig.add_subplot(247)
        mask_3c = np.stack([mask_crop] * 3, axis=2)
        ax7.imshow(mask_3c)

        ax8 = fig.add_subplot(248)
        ax8.imshow((mask_3c * image_v).astype(np.uint8))

        plt.show()
