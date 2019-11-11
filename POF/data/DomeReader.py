import tensorflow as tf
from data.BaseReader import BaseReader
import numpy as np
import numpy.linalg as nl
import pickle
from utils.keypoint_conversion import a4_to_main as order_dict
import json
import os


class DomeReader(BaseReader):

    def __init__(self, mode='training', objtype=0, shuffle=False, batch_size=1, crop_noise=False, full_only=True, head_top=True):
        super(DomeReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode in ('training', 'evaluation')
        self.image_root = '/media/posefs0c/panopticdb/'

        # read data from a4
        path_to_db = './data/a4_collected.pkl'
        path_to_calib = './data/camera_data_a4.pkl'

        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

        with open('./data/a4_hands_annotated.txt') as f:
            hand_annots = {}
            for line in f:
                strs = line.split()
                hand_annots[tuple(strs[:3])] = eval(strs[3])

        if mode == 'training':
            mode_data = db_data['training_data']
        else:
            mode_data = db_data['testing_data']

        with open(path_to_calib, 'rb') as f:
            calib_data = pickle.load(f)

        human3d = {'body': [], 'left_hand': [], 'right_hand': [], 'body_valid': [], 'left_hand_valid': [], 'right_hand_valid': []}
        calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
        img_dirs = []

        for data3d in mode_data:
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']

            # check for manual annotation, remove the annotation if the hand is annotated as incorrect.
            if 'left_hand' in data3d and not hand_annots[(seqName, frame_str, 'left')]:
                del data3d['left_hand']
            if 'right_hand' in data3d and not hand_annots[(seqName, frame_str, 'righ')]:
                del data3d['right_hand']

            if objtype == 0:
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                nose_lear = body3d[16] - body3d[1]
                nose_rear = body3d[18] - body3d[1]
                neck_nose = body3d[1] - body3d[0]
                n = np.cross(nose_lear, nose_rear)
                n = n / nl.norm(n)
                d = np.dot(neck_nose, n)
                assert d > 0
                head_top_kp = body3d[0] + 1.5 * d * n
                if head_top:
                    body3d = np.concatenate((body3d, head_top_kp[np.newaxis, :]), axis=0)
                chest = 0.5 * body3d[0] + 0.25 * (body3d[6] + body3d[12])
                body3d = np.concatenate((body3d, chest[np.newaxis, :]), axis=0)

            elif objtype == 1:
                # left hand or right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if ('left_hand' not in data3d) and ('right_hand' not in data3d):
                    continue

            else:
                assert objtype == 2
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                # both left and right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                    # discard the sample if hand is wanted but there is no left hand.
                else:
                    continue
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                else:
                    continue

            if objtype == 0:
                for camIdx, camDict in data3d['body']['2D'].items():
                    if full_only:
                        cond_inside = all(camDict['insideImg'])
                    else:  # if not full_only, use the image if at least half keypoints are visible
                        inside_ratio = np.float(np.sum(camDict['insideImg'])) / len(camDict['insideImg'])
                        cond_inside = (inside_ratio > 0.5)
                    if any(camDict['occluded']) or not cond_inside:
                        continue
                    human3d['body'].append(body3d)
                    human3d['body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

            elif objtype == 1:
                if 'left_hand' in data3d:
                    for camIdx, camDict in data3d['left_hand']['2D'].items():
                        if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']) or data3d['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['left_hand'].append(left_hand3d)
                        human3d['right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

                if 'right_hand' in data3d:
                    for camIdx, camDict in data3d['right_hand']['2D'].items():
                        if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']) or data3d['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['right_hand'].append(right_hand3d)
                        human3d['left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

            else:
                assert objtype == 2
                for camIdx, camDict in data3d['body']['2D'].items():
                    if any(camDict['occluded']) or not all(camDict['insideImg']):
                        continue
                    if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']):
                        continue
                    if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']):
                        continue
                    # If this line is reached, the sample and cam view is valid.
                    human3d['body'].append(body3d)
                    human3d['left_hand'].append(left_hand3d)
                    human3d['right_hand'].append(right_hand3d)
                    human3d['body_valid'].append(np.ones((18,), dtype=bool))
                    human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                    human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

        if mode == 'evaluation':
            if objtype == 2:
                openpose_output_file = '/home/donglaix/Documents/Experiments/dome_valid/a4_openpose.json'
                assert os.path.exists(openpose_output_file)
                with open(openpose_output_file) as f:
                    openpose_data = json.load(f)
                openpose_data = np.array(openpose_data, dtype=np.float32).reshape(-1, 70, 3)
                openpose_valid = (openpose_data[:, :, 2] >= 0.5)
                openpose_data[:, :, 0] *= openpose_valid
                openpose_data[:, :, 1] *= openpose_valid
                openpose_face = openpose_data[:, :, :2]
                human3d['openpose_face'] = openpose_face

        # read data from a5
        path_to_db = './data/a5_collected.pkl'
        path_to_calib = './data/camera_data_a5.pkl'

        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

        if mode == 'training':
            mode_data = db_data['training_data']
        else:
            mode_data = db_data['testing_data']

        with open(path_to_calib, 'rb') as f:
            calib_data = pickle.load(f)

        for data3d in mode_data:
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']

            if objtype == 0:
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                nose_lear = body3d[16] - body3d[1]
                nose_rear = body3d[18] - body3d[1]
                neck_nose = body3d[1] - body3d[0]
                n = np.cross(nose_lear, nose_rear)
                n = n / nl.norm(n)
                d = np.dot(neck_nose, n)
                assert d > 0
                if head_top:
                    head_top_kp = body3d[0] + 1.5 * d * n
                    body3d = np.concatenate((body3d, head_top_kp[np.newaxis, :]), axis=0)
                chest = 0.5 * body3d[0] + 0.25 * (body3d[6] + body3d[12])
                body3d = np.concatenate((body3d, chest[np.newaxis, :]), axis=0)

            elif objtype == 1:
                # left hand or right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if ('left_hand' not in data3d) and ('right_hand' not in data3d):
                    continue

            else:
                assert objtype == 2
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                # both left and right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                    # discard the sample if hand is wanted but there is no left hand.
                else:
                    continue
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                else:
                    continue

            if objtype == 0:
                for camIdx, camDict in data3d['body']['2D'].items():
                    if full_only:
                        cond_inside = all(camDict['insideImg'])
                    else:  # if not full_only, use the image if at least half keypoints are visible
                        inside_ratio = np.float(np.sum(camDict['insideImg'])) / len(camDict['insideImg'])
                        cond_inside = (inside_ratio > 0.5)
                    if any(camDict['occluded']) or not cond_inside:
                        continue
                    human3d['body'].append(body3d)
                    human3d['body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

            elif objtype == 1:
                if 'left_hand' in data3d:
                    for camIdx, camDict in data3d['left_hand']['2D'].items():
                        if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']) or data3d['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['left_hand'].append(left_hand3d)
                        human3d['right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

                if 'right_hand' in data3d:
                    for camIdx, camDict in data3d['right_hand']['2D'].items():
                        if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']) or data3d['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['right_hand'].append(right_hand3d)
                        human3d['left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

            else:
                assert objtype == 2
                for camIdx, camDict in data3d['body']['2D'].items():
                    if any(camDict['occluded']) or not all(camDict['insideImg']):
                        continue
                    if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']):
                        continue
                    if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']):
                        continue
                    # If this line is reached, the sample and cam view is valid.
                    human3d['body'].append(body3d)
                    human3d['left_hand'].append(left_hand3d)
                    human3d['right_hand'].append(right_hand3d)
                    human3d['body_valid'].append(np.ones((18,), dtype=bool))
                    human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                    human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

        human3d.update(calib)
        human3d['img_dirs'] = img_dirs

        # import cv2
        # for img_dir in img_dirs:
        #     if cv2.imread(img_dir) is None:
        #         print(img_dir)

        self.register_tensor(human3d, order_dict)
        self.num_samples = len(self.tensor_dict['img_dirs'])

    def get(self, withPAF=True):
        d = super(DomeReader, self).get(withPAF=withPAF)
        return d


if __name__ == '__main__':
    d = DomeReader(mode='training', shuffle=True, objtype=1, crop_noise=True, full_only=False)
    # d.rotate_augmentation = True
    # d.blur_augmentation = True
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
                      data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_xyz_local']])
        image_crop, img_dir, hand2d, hand_valid, hand2d_heatmap, PAF, mask_crop, hand3d = [np.squeeze(_) for _ in values]

        image_name = img_dir.item().decode()
        image_v = ((image_crop + 0.5) * 255).astype(np.uint8)

        hand2d_detected, bscore = utils.PAF.detect_keypoints2d_PAF(hand2d_heatmap, PAF, objtype=1)
        # hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:20, :]
        hand3d_detected, _ = PAF_to_3D(hand2d_detected, PAF, objtype=1)
        hand3d_detected = hand3d_detected[:21, :]

        fig = plt.figure(1)
        ax1 = fig.add_subplot(231)
        plt.imshow(image_v)
        utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

        ax2 = fig.add_subplot(232, projection='3d')
        utils.general.plot3d(ax2, hand3d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.axis('equal')

        ax3 = fig.add_subplot(233, projection='3d')
        utils.general.plot3d(ax3, hand3d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        plt.axis('equal')

        xy, z = plot_all_PAF(PAF, 3)
        ax4 = fig.add_subplot(234)
        ax4.imshow(xy)

        ax5 = fig.add_subplot(235)
        ax5.imshow(z)

        plt.show()
