import tensorflow as tf
from data.BaseReader import BaseReader
import numpy as np
import pickle
from utils.keypoint_conversion import a4_to_main as order_dict
import json
import os


class OpenposeReader(BaseReader):

    def __init__(self, seqName, mode='evaluation', objtype=0, shuffle=False, batch_size=1, crop_noise=False):
        super(OpenposeReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode == 'evaluation'
        assert objtype == 2
        self.image_root = '/media/posefs1b/Users/donglaix/siggasia018/{}/'.format(seqName)
        assert os.path.isdir(self.image_root)

        path_to_db = './data/{}.pkl'.format(seqName)

        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

        human3d = {}
        num_samples = len(db_data[0])
        K = np.array(db_data[5]['K'], dtype=np.float32)
        K = np.expand_dims(K, axis=0)
        K = np.tile(K, (num_samples, 1, 1))
        human3d['K'] = K
        human3d['openpose_body'] = db_data[0].astype(np.float32)[:, :18, :]
        # duplicate the neck for head top and chest
        human3d['openpose_body'] = np.concatenate((human3d['openpose_body'], human3d['openpose_body'][:, 1:2, :], human3d['openpose_body'][:, 1:2, :]), axis=1)
        human3d['openpose_body_score'] = db_data[0][:, :18, 2].astype(np.float32)
        # duplicate the neck for head top and chest
        human3d['openpose_body_score'] = np.concatenate((human3d['openpose_body_score'], human3d['openpose_body_score'][:, 1:2], human3d['openpose_body_score'][:, 1:2]), axis=1)
        human3d['openpose_lhand'] = db_data[1].astype(np.float32)
        human3d['openpose_lhand_score'] = db_data[1][:, :, 2].astype(np.float32)
        human3d['openpose_rhand'] = db_data[2].astype(np.float32)
        human3d['openpose_rhand_score'] = db_data[2][:, :, 2].astype(np.float32)
        human3d['openpose_face'] = db_data[3].astype(np.float32)
        human3d['openpose_face_score'] = db_data[3][:, :, 2].astype(np.float32)
        human3d['openpose_foot'] = db_data[0].astype(np.float32)[:, 18:, :]
        human3d['openpose_foot_score'] = db_data[0].astype(np.float32)[:, 18:, 2]
        human3d['img_dirs'] = np.core.defchararray.add(np.array([self.image_root]), db_data[4])

        human3d['body_valid'] = np.ones((num_samples, 20), dtype=bool)
        human3d['left_hand_valid'] = np.ones((num_samples, 21), dtype=bool)
        human3d['right_hand_valid'] = np.ones((num_samples, 21), dtype=bool)

        # dummy values
        R = np.eye(3, dtype=np.float32)
        R = np.expand_dims(R, axis=0)
        R = np.tile(R, (num_samples, 1, 1))
        human3d['R'] = R
        t = np.ones((3,), dtype=np.float32)
        t = np.expand_dims(t, axis=0)
        t = np.tile(t, (num_samples, 1))
        human3d['t'] = t
        dc = np.zeros((5,), dtype=np.float32)
        dc = np.expand_dims(dc, axis=0)
        dc = np.tile(dc, (num_samples, 1))
        human3d['distCoef'] = dc

        human3d['body'] = np.zeros((num_samples, 21, 3), dtype=np.float32)
        human3d['left_hand'] = np.zeros((num_samples, 21, 3), dtype=np.float32)
        human3d['right_hand'] = np.zeros((num_samples, 21, 3), dtype=np.float32)

        for key, val in human3d.items():
            if 'openpose' in key and 'score' not in key:
                # valid = val[:, :, 2] > 0.05
                valid = val[:, :, 2] > 0.0
                val[:, :, 0] *= valid
                val[:, :, 1] *= valid
                human3d[key] = val[:, :, :2]

        self.register_tensor(human3d, order_dict)
        self.num_samples = len(self.tensor_dict['img_dirs'])

    def get(self, imw=1920, imh=1080):
        d = super(OpenposeReader, self).get(withPAF=False, bbox2d=1, imw=imw, imh=imh)
        return d


if __name__ == '__main__':
    d = OpenposeReader(mode='evaluation', seqName='test3', shuffle=False, objtype=2, crop_noise=False)
    data_dict = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import utils.general
    from utils.vis_heatmap3d import vis_heatmap3d

    validation_images = []

    for i in range(d.num_samples):
        print('{}/{}'.format(i + 1, d.num_samples))
        bimage_crop, image, body2d, body2d_local, foot2d = sess.run([data_dict['bimage_crop'], data_dict['image'], data_dict['openpose_body'], data_dict['body_uv_local'], data_dict['openpose_foot']])
        foot2d = np.squeeze(foot2d)
        image_v = ((image[0] + 0.5) * 255).astype(np.uint8)
        image_crop_v = ((bimage_crop[0] + 0.5) * 255).astype(np.uint8)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        plt.imshow(image_v)
        plt.scatter(foot2d[:, 0], foot2d[:, 1])
        for i in range(4):
            plt.text(int(foot2d[i, 0]), int(foot2d[i, 1]), str(i))
        utils.general.plot2d(ax1, body2d[0])

        ax2 = fig.add_subplot(122)
        plt.imshow(image_crop_v)
        utils.general.plot2d(ax2, body2d_local[0])
        plt.show()
