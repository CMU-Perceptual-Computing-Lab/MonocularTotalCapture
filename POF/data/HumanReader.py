import tensorflow as tf
from data.BaseReader import BaseReader
import numpy as np
import h5py
from utils.keypoint_conversion import human36m_to_main, mpi3d_to_main, SMPL_to_main
import pickle
import os


class HumanReader(BaseReader):

    def __init__(self, name='Human3.6M', mode='training', objtype=0, shuffle=True, batch_size=1, crop_noise=False):
        super(HumanReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert objtype == 0  # this dataset reader only supports human body data
        assert mode in ('training', 'evaluation')
        self.mode = mode
        assert name in ('Human3.6M', 'MPI_INF_3DHP', 'UP', 'SURREAL')
        self.name = name

        if name == 'Human3.6M':
            self.image_root = '/media/posefs1b/Users/donglaix/c2f-vol-train/data/h36m/images/'

            if self.mode == 'training':
                image_list_file = '/media/posefs1b/Users/donglaix/c2f-vol-train/data/h36m/annot/train_images.txt'
                path_to_db = '/media/posefs1b/Users/donglaix/c2f-vol-train/data/h36m/annot/train.h5'
            else:
                image_list_file = '/media/posefs1b/Users/donglaix/c2f-vol-train/data/h36m/annot/valid_images.txt'
                path_to_db = '/media/posefs1b/Users/donglaix/c2f-vol-train/data/h36m/annot/valid.h5'

            path_to_calib = '/media/posefs3b/Users/donglaix/h36m/cameras.h5'

            with open(image_list_file) as f:
                img_list = [_.strip() for _ in f]
            fannot = h5py.File(path_to_db, 'r')
            annot3d = fannot['S'][:]
            annot2d = fannot['part'][:]
            fannot.close()
            fcalib = h5py.File(path_to_calib, 'r')
            calib_data = {}
            map_camera = {'54138969': 'camera1', '55011271': 'camera2', '58860488': 'camera3', '60457274': 'camera4'}
            for pid in fcalib.keys():
                if pid == '3dtest':
                    continue
                person_cam_data = {}
                for camera in map_camera.values():
                    cam_data = {_: fcalib[pid][camera][_][:] for _ in fcalib[pid][camera].keys()}
                    person_cam_data[camera] = cam_data
                calib_data[pid] = person_cam_data
            fcalib.close()

            human3d = {'body': [], 'left_hand': [], 'right_hand': [], 'gt_body': []}
            calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
            img_dirs = []

            for img_idx, img_name in enumerate(img_list):
                img_dir = os.path.join(self.image_root, img_name)

                body2d = annot2d[img_idx].astype(np.float32)
                if mode == 'training' and (body2d >= 1000).any() or (body2d <= 0).any():
                    continue
                body3d = annot3d[img_idx].astype(np.float32)
                human3d['gt_body'].append(body3d)
                body3d = np.concatenate((body3d, np.ones((1, 3), dtype=np.float32)), axis=0)  # put dummy values in order_dict

                person = img_name.split('_')[0].replace('S', 'subject')
                camera = img_name.split('.')[1].split('_')[0]
                camera_name = map_camera[camera]
                cam_param = calib_data[person][camera_name]

                K = np.eye(3)
                K[0, 0] = cam_param['f'][0, 0]
                K[1, 1] = cam_param['f'][1, 0]
                K[0, 2] = cam_param['c'][0, 0]
                K[1, 2] = cam_param['c'][1, 0]
                dc = np.zeros((5,))
                dc[:3] = cam_param['k'][:, 0]
                dc[3:] = cam_param['p'][:, 0]

                human3d['body'].append(body3d)
                img_dirs.append(img_dir)

                calib['K'].append(K.astype(np.float32))
                calib['R'].append(np.eye(3, dtype=np.float32))
                calib['t'].append(np.zeros((3,), dtype=np.float32))
                calib['distCoef'].append(dc.astype(np.float32))

            self.num_samples = len(img_dirs)

            human3d.update(calib)
            human3d['img_dirs'] = img_dirs
            body_valid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)
            human3d['body_valid'] = np.tile(body_valid, (self.num_samples, 1))
            order_dict = human36m_to_main

        elif name == 'MPI_INF_3DHP':
            self.image_root = '/media/posefs1b/Users/donglaix/mpi_inf_3dhp/'
            assert mode == 'training'
            self.path_to_db = self.image_root + 'mpi3d.pickle'
            with open(self.path_to_db, 'rb') as f:
                db_data = pickle.load(f)
            img_dirs, body, K = db_data
            self.num_samples = img_dirs.shape[0]
            img_dirs = np.core.defchararray.add(np.array([self.image_root]), img_dirs)
            body = body.astype(np.float32)
            body = np.concatenate([body, np.ones((self.num_samples, 1, 3))], axis=1).astype(np.float32)
            K = K.astype(np.float32)
            body_valid = np.ones((self.num_samples, 19), dtype=bool)
            body_valid[:, 0] = False
            body_valid[:, 14:18] = False
            R = np.tile(np.expand_dims(np.eye(3, dtype=np.float32), axis=0), (self.num_samples, 1, 1))
            t = np.tile(np.zeros((1, 3), dtype=np.float32), (self.num_samples, 1))
            dc = np.tile(np.zeros((1, 5), dtype=np.float32), (self.num_samples, 1))
            human3d = {'img_dirs': img_dirs, 'body': body, 'K': K, 'body_valid': body_valid, 'R': R, 't': t, 'distCoef': dc}
            order_dict = mpi3d_to_main

        elif name == 'UP':
            self.image_root = '/media/posefs3b/Users/donglaix/UP/'
            assert mode in 'training'
            self.path_to_db = './data/UP_collected.pkl'
            with open(self.path_to_db, 'rb') as f:
                db_data = pickle.load(f, encoding='latin')
            human3d = {'body': [], 'img_dirs': [], 'body_valid': [], 'mask_dirs': []}
            calib = {'K': [], 'R': [], 't': [], 'distCoef': []}

            for data in db_data:
                calib['K'].append(data['K'].astype(np.float32))
                calib['R'].append(data['R'].astype(np.float32))
                calib['t'].append(data['t'].astype(np.float32))
                calib['distCoef'].append(np.zeros([5], dtype=np.float32))
                human3d['body'].append(data['J'].astype(np.float32))
                body_valid = np.ones([19], dtype=bool)
                # body_valid[0] = False
                # body_valid[14:] = False
                human3d['body_valid'].append(body_valid)
                human3d['img_dirs'].append(os.path.join(self.image_root, data['img_dir']))
                human3d['mask_dirs'].append(os.path.join(self.image_root, data['mask']))

            human3d.update(calib)
            order_dict = SMPL_to_main
            self.num_samples = len(human3d['img_dirs'])

        elif name == 'SURREAL':
            self.image_root = '/media/posefs3b/Users/donglaix/surreal/surreal/SURREAL/'
            assert mode in 'training'
            self.path_to_db = os.path.join(self.image_root, 'surreal_collected.pkl')
            with open(self.path_to_db, 'rb') as f:
                db_data = pickle.load(f, encoding='latin')
            human3d = {'body': [], 'img_dirs': [], 'body_valid': []}
            calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
            for data in db_data:
                calib['K'].append(data['K'].astype(np.float32))
                calib['R'].append(data['R'].astype(np.float32))
                calib['t'].append(np.ravel(data['t']).astype(np.float32))
                calib['distCoef'].append(np.zeros([5], dtype=np.float32))
                human3d['body'].append(data['J'].astype(np.float32))
                body_valid = np.ones([19], dtype=bool)
                body_valid[0] = False
                body_valid[14:] = False
                human3d['body_valid'].append(body_valid)
                human3d['img_dirs'].append(os.path.join(self.image_root, data['img_dir']))

            human3d.update(calib)
            order_dict = SMPL_to_main
            self.num_samples = len(human3d['img_dirs'])

        else:
            raise NotImplementedError

        self.register_tensor(human3d, order_dict)

    def get(self):
        if self.name == 'Human3.6M':
            d = super(HumanReader, self).get(withPAF=True, imw=1000, imh=1002)
        elif self.name == 'MPI_INF_3DHP':
            d = super(HumanReader, self).get(withPAF=True, imw=2048, imh=2048)
        elif self.name == 'UP':
            d = super(HumanReader, self).get(withPAF=True, imw=1920, imh=1080)
        elif self.name == 'SURREAL':
            d = super(HumanReader, self).get(withPAF=True, imw=320, imh=240)
        else:
            raise NotImplementedError
        return d


if __name__ == '__main__':
    d = HumanReader(mode='evaluation', name='Human3.6M', shuffle=False, objtype=0, crop_noise=False)
    d.start_from(77095)
    # d.crop_size_zoom = 1.5
    data_dict = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import utils.general
    from utils.PAF import plot_all_PAF
    # from utils.vis_heatmap3d import vis_heatmap3d

    validation_images = []

    for i in range(d.num_samples):
        print('{}/{}'.format(i + 1, d.num_samples))
        bimage_crop, img_dir, body2d, body_valid, body2d_heatmap, body3d, PAF, mask_crop = \
            sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['body_valid'], data_dict['scoremap2d'],
                      data_dict['keypoint_xyz_local'], data_dict['PAF'], data_dict['mask_crop']])
        image_name = img_dir[0].decode()
        print(image_name)
        image_v = ((bimage_crop[0] + 0.5) * 255).astype(np.uint8)
        body2d = np.squeeze(body2d)
        body_valid = np.squeeze(body_valid)
        body2d_heatmap = np.squeeze(body2d_heatmap)
        body3d = np.squeeze(body3d)
        mask_crop = np.squeeze(mask_crop).astype(bool)
        PAF = np.squeeze(PAF)

        body2d_detected = utils.general.detect_keypoints2d(body2d_heatmap)[:19, :]

        fig = plt.figure(1)
        ax1 = fig.add_subplot(161)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, body2d, valid_idx=body_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, body2d_detected, valid_idx=body_valid, color=np.array([0.0, 0.0, 1.0]))
        for i in range(19):
            plt.text(int(body2d[i, 0]), int(body2d[i, 1]), str(i))

        ax2 = fig.add_subplot(162, projection='3d')
        utils.general.plot3d(ax2, body3d, valid_idx=body_valid, color=np.array([1.0, 0.0, 0.0]))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        ax2.set_xlim(0, 47)
        ax2.set_ylim(0, 47)
        ax2.set_zlim(0, 47)

        ax3 = fig.add_subplot(163)
        plt.imshow(mask_crop)

        ax4 = fig.add_subplot(164)
        mask_3c = np.stack([mask_crop] * 3, axis=2)
        masked = mask_3c * image_v
        plt.imshow(masked)

        xy, z = plot_all_PAF(PAF, 3)
        ax5 = fig.add_subplot(165)
        ax5.imshow(xy)

        ax6 = fig.add_subplot(166)
        ax6.imshow(z)

        plt.show()
