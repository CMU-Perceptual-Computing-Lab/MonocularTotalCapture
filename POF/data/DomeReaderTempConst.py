import tensorflow as tf
from data.TempConstReader import TempConstReader
import numpy as np
import numpy.linalg as nl
import pickle
from utils.keypoint_conversion import a4_to_main as order_dict
import json
import os


class DomeReaderTempConst(TempConstReader):

    def __init__(self, mode='training', objtype=0, shuffle=False, batch_size=1, crop_noise=False, full_only=True, head_top=True):
        super(DomeReaderTempConst, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert mode in ('training', 'evaluation')
        assert objtype in (0, 1)
        self.image_root = '/media/posefs0c/panopticdb/'

        # read data from a4plus with consecutive frames
        path_to_db = './data/a4plus_collected.pkl'
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

        human3d = {'1_body': [], '1_left_hand': [], '1_right_hand': [], '1_body_valid': [], 'left_hand_valid': [], 'right_hand_valid': [],
                   '2_body': [], '2_left_hand': [], '2_right_hand': [], '2_body_valid': []}
        calib = {'1_K': [], '1_R': [], '1_t': [], '1_distCoef': [],
                 '2_K': [], '2_R': [], '2_t': [], '2_distCoef': []}
        img_dirs_1 = []
        img_dirs_2 = []

        map_next = {}
        for data3d in mode_data:
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']
            frame = int(frame_str)
            if frame % 5:  # a4plus is sampled 1 out of 5, frame number *0 and *5 is the first frame, *1 and *6 is the second frame
                continue
            map_next[(seqName, frame_str)] = None
        for data3d in mode_data:
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']
            frame = int(frame_str)
            if frame % 5 != 1:
                continue
            prev_key = (seqName, '{:08d}'.format(frame - 1))
            if prev_key not in map_next:
                continue
            map_next[prev_key] = data3d

        for data3d in mode_data:
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']
            frame = int(frame_str)
            if frame % 5:
                continue

            # check for manual annotation, remove the annotation if the hand is annotated as incorrect.
            if 'left_hand' in data3d and not hand_annots[(seqName, frame_str, 'left')]:
                del data3d['left_hand']
            if 'right_hand' in data3d and not hand_annots[(seqName, frame_str, 'righ')]:
                del data3d['right_hand']

            next_data = map_next[(seqName, frame_str)]
            if next_data is None:
                continue

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
                body3d_1 = np.concatenate((body3d, chest[np.newaxis, :]), axis=0)

                body3d = np.array(next_data['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
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
                body3d_2 = np.concatenate((body3d, chest[np.newaxis, :]), axis=0)

            elif objtype == 1:
                # left hand or right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d_1 = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'left_hand' in next_data:
                    left_hand3d_2 = np.array(next_data['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in data3d:
                    right_hand3d_1 = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in next_data:
                    right_hand3d_2 = np.array(next_data['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if ('left_hand' not in data3d or 'left_hand' not in next_data) and ('right_hand' not in data3d or 'right_hand' not in next_data):
                    # one hand must be valid for both frames
                    continue

            if objtype == 0:
                for camIdx, camDict in data3d['body']['2D'].items():
                    if camIdx not in next_data['body']['2D']:  # no data from this camera in the next frame
                        continue
                    if full_only:
                        cond_inside_1 = all(camDict['insideImg'])
                        cond_inside_2 = all(next_data['body'][camIdx]['insideImg'])
                    else:  # if not full_only, use the image if at least half keypoints are visible
                        inside_ratio_1 = np.float(np.sum(camDict['insideImg'])) / len(camDict['insideImg'])
                        inside_ratio_2 = np.float(np.sum(next_data['body']['2D'][camIdx]['insideImg'])) / len(next_data['body']['2D'][camIdx]['insideImg'])
                        cond_inside_1 = (inside_ratio_1 > 0.1)
                        cond_inside_2 = (inside_ratio_2 > 0.1)
                    if any(camDict['occluded']) or any(next_data['body']['2D'][camIdx]['occluded']) or not cond_inside_1 or not cond_inside_2:
                        continue
                    human3d['1_body'].append(body3d_1)
                    human3d['2_body'].append(body3d_2)
                    human3d['1_body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    human3d['2_body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    calib['1_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['2_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['1_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['2_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['1_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['2_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['1_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    calib['2_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs_1.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))
                    img_dirs_2.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, next_data['frame_str'], camIdx, next_data['frame_str']))

            elif objtype == 1:
                if 'left_hand' in data3d and 'left_hand' in next_data:
                    for camIdx, camDict in data3d['left_hand']['2D'].items():
                        if camIdx not in next_data['left_hand']['2D']:
                            continue
                        if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']) or data3d['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        if any(next_data['left_hand']['2D'][camIdx]['occluded']) or not all(next_data['left_hand']['2D'][camIdx]['insideImg']) or next_data['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['1_left_hand'].append(left_hand3d_1)
                        human3d['2_left_hand'].append(left_hand3d_2)
                        human3d['1_right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['2_right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))
                        calib['1_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['2_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['1_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['2_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['1_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['2_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['1_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        calib['2_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs_1.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))
                        img_dirs_2.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, next_data['frame_str'], camIdx, next_data['frame_str']))

                if 'right_hand' in data3d and 'right_hand' in next_data:
                    for camIdx, camDict in data3d['right_hand']['2D'].items():
                        if camIdx not in next_data['right_hand']['2D']:
                            continue
                        if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']) or data3d['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        if any(next_data['right_hand']['2D'][camIdx]['occluded']) or not all(next_data['right_hand']['2D'][camIdx]['insideImg']) or next_data['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['1_right_hand'].append(right_hand3d_1)
                        human3d['2_right_hand'].append(right_hand3d_2)
                        human3d['1_left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['2_left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                        calib['1_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['2_K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['1_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['2_R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['1_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['2_t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['1_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        calib['2_distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs_1.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))
                        img_dirs_2.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, next_data['frame_str'], camIdx, next_data['frame_str']))

        human3d.update(calib)
        human3d['1_img_dirs'] = img_dirs_1
        human3d['2_img_dirs'] = img_dirs_2

        self.register_tensor(human3d, order_dict)
        self.num_samples = len(self.tensor_dict['1_img_dirs'])

    def get(self, withPAF=True):
        d = super(DomeReaderTempConst, self).get(withPAF=withPAF)
        return d


if __name__ == '__main__':
    # d = DomeReaderTempConst(mode='training', shuffle=True, objtype=0, crop_noise=True, full_only=False)
    # data_dict = d.get()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess.run(tf.global_variables_initializer())
    # tf.train.start_queue_runners(sess=sess)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import utils.general
    # from utils.vis_heatmap3d import vis_heatmap3d
    # from utils.PAF import plot_PAF, PAF_to_3D, plot_all_PAF

    # validation_images = []

    # for i in range(d.num_samples):
    #     print('{}/{}'.format(i + 1, d.num_samples))
    #     values = \
    #         sess.run([data_dict['1_image_crop'], data_dict['1_img_dir'], data_dict['1_keypoint_uv_local'], data_dict['1_body_valid'], data_dict['1_scoremap2d'],
    #                   data_dict['1_PAF'], data_dict['1_mask_crop'], data_dict['1_keypoint_xyz_local'],
    #                   data_dict['2_image_crop'], data_dict['2_img_dir'], data_dict['2_keypoint_uv_local'], data_dict['2_body_valid'], data_dict['2_scoremap2d'],
    #                   data_dict['2_PAF'], data_dict['2_mask_crop'], data_dict['2_keypoint_xyz_local']
    #                   ])
    #     image_crop_1, img_dir_1, body2d_1, body_valid_1, body2d_heatmap_1, PAF_1, mask_crop_1, body3d_1, \
    #         image_crop_2, img_dir_2, body2d_2, body_valid_2, body2d_heatmap_2, PAF_2, mask_crop_2, body3d_2 = [np.squeeze(_) for _ in values]

    #     image_name_1 = img_dir_1.item().decode()
    #     image_name_2 = img_dir_2.item().decode()
    #     image_v_1 = ((image_crop_1 + 0.5) * 255).astype(np.uint8)
    #     image_v_2 = ((image_crop_2 + 0.5) * 255).astype(np.uint8)

    #     body2d_detected_1, bscore_1 = utils.PAF.detect_keypoints2d_PAF(body2d_heatmap_1, PAF_1)
    #     body2d_detected_2, bscore_2 = utils.PAF.detect_keypoints2d_PAF(body2d_heatmap_2, PAF_2)
    #     # body2d_detected = utils.general.detect_keypoints2d(body2d_heatmap)[:20, :]
    #     body3d_detected_1, _ = PAF_to_3D(body2d_detected_1, PAF_1, objtype=0)
    #     body3d_detected_2, _ = PAF_to_3D(body2d_detected_2, PAF_2, objtype=0)
    #     # body3d_detected = body3d_detected[:21, :]

    #     fig = plt.figure(1)
    #     ax1 = fig.add_subplot(241)
    #     plt.imshow(image_v_1)
    #     utils.general.plot2d(ax1, body2d_1, type_str='body', valid_idx=body_valid_1, color=np.array([1.0, 0.0, 0.0]))
    #     utils.general.plot2d(ax1, body2d_detected_1, type_str='body', valid_idx=body_valid_1, color=np.array([0.0, 0.0, 1.0]))

    #     ax2 = fig.add_subplot(242)
    #     plt.imshow(image_v_2)
    #     utils.general.plot2d(ax2, body2d_2, type_str='body', valid_idx=body_valid_2, color=np.array([1.0, 0.0, 0.0]))
    #     utils.general.plot2d(ax2, body2d_detected_2, type_str='body', valid_idx=body_valid_2, color=np.array([0.0, 0.0, 1.0]))

    #     ax3 = fig.add_subplot(243, projection='3d')
    #     utils.general.plot3d(ax3, body3d_detected_1, type_str='body', valid_idx=body_valid_1, color=np.array([0.0, 0.0, 1.0]))
    #     utils.general.plot3d(ax3, body3d_detected_2, type_str='body', valid_idx=body_valid_2, color=np.array([0.0, 0.0, 1.0]))
    #     ax3.set_xlabel('X Label')
    #     ax3.set_ylabel('Y Label')
    #     ax3.set_zlabel('Z Label')
    #     plt.axis('equal')

    #     ax4 = fig.add_subplot(244, projection='3d')
    #     utils.general.plot3d(ax4, body3d_1, type_str='body', valid_idx=body_valid_1, color=np.array([1.0, 0.0, 0.0]))
    #     utils.general.plot3d(ax4, body3d_2, type_str='body', valid_idx=body_valid_2, color=np.array([1.0, 0.0, 0.0]))
    #     ax4.set_xlabel('X Label')
    #     ax4.set_ylabel('Y Label')
    #     ax4.set_zlabel('Z Label')
    #     plt.axis('equal')

    #     xy, z = plot_all_PAF(PAF_1, 3)
    #     ax5 = fig.add_subplot(245)
    #     ax5.imshow(xy)

    #     ax6 = fig.add_subplot(246)
    #     ax6.imshow(z)

    #     xy, z = plot_all_PAF(PAF_2, 3)
    #     ax7 = fig.add_subplot(247)
    #     ax7.imshow(xy)

    #     ax8 = fig.add_subplot(248)
    #     ax8.imshow(z)

    #     plt.show()

    d = DomeReaderTempConst(mode='training', shuffle=True, objtype=1, crop_noise=True, full_only=False)
    d.crop_scale_noise_sigma = 0.4
    d.crop_offset_noise_sigma = 0.2
    data_dict = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
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
            sess.run([data_dict['1_image_crop'], data_dict['1_img_dir'], data_dict['1_keypoint_uv_local'], data_dict['1_hand_valid'], data_dict['1_scoremap2d'],
                      data_dict['1_PAF'], data_dict['1_mask_crop'], data_dict['1_keypoint_xyz_local'],
                      data_dict['2_image_crop'], data_dict['2_img_dir'], data_dict['2_keypoint_uv_local'], data_dict['2_hand_valid'], data_dict['2_scoremap2d'],
                      data_dict['2_PAF'], data_dict['2_mask_crop'], data_dict['2_keypoint_xyz_local']
                      ])
        image_crop_1, img_dir_1, body2d_1, body_valid_1, body2d_heatmap_1, PAF_1, mask_crop_1, body3d_1, \
            image_crop_2, img_dir_2, body2d_2, body_valid_2, body2d_heatmap_2, PAF_2, mask_crop_2, body3d_2 = [np.squeeze(_) for _ in values]

        image_name_1 = img_dir_1.item().decode()
        image_name_2 = img_dir_2.item().decode()
        image_v_1 = ((image_crop_1 + 0.5) * 255).astype(np.uint8)
        image_v_2 = ((image_crop_2 + 0.5) * 255).astype(np.uint8)

        body2d_detected_1, bscore_1 = utils.PAF.detect_keypoints2d_PAF(body2d_heatmap_1, PAF_1, objtype=1)
        body2d_detected_2, bscore_2 = utils.PAF.detect_keypoints2d_PAF(body2d_heatmap_2, PAF_2, objtype=1)
        # body2d_detected = utils.general.detect_keypoints2d(body2d_heatmap)[:20, :]
        body3d_detected_1, _ = PAF_to_3D(body2d_detected_1, PAF_1, objtype=1)
        body3d_detected_2, _ = PAF_to_3D(body2d_detected_2, PAF_2, objtype=1)
        body3d_detected_1 = body3d_detected_1[:21, :]
        body3d_detected_2 = body3d_detected_2[:21, :]

        fig = plt.figure(1)
        ax1 = fig.add_subplot(241)
        plt.imshow(image_v_1)
        utils.general.plot2d(ax1, body2d_1, type_str='hand', valid_idx=body_valid_1, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax1, body2d_detected_1, type_str='hand', valid_idx=body_valid_1, color=np.array([0.0, 0.0, 1.0]))

        ax2 = fig.add_subplot(242)
        plt.imshow(image_v_2)
        utils.general.plot2d(ax2, body2d_2, type_str='hand', valid_idx=body_valid_2, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot2d(ax2, body2d_detected_2, type_str='hand', valid_idx=body_valid_2, color=np.array([0.0, 0.0, 1.0]))

        ax3 = fig.add_subplot(243, projection='3d')
        utils.general.plot3d(ax3, body3d_detected_1, type_str='hand', valid_idx=body_valid_1, color=np.array([0.0, 0.0, 1.0]))
        utils.general.plot3d(ax3, body3d_detected_2, type_str='hand', valid_idx=body_valid_2, color=np.array([0.0, 0.0, 1.0]))
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        plt.axis('equal')

        ax4 = fig.add_subplot(244, projection='3d')
        utils.general.plot3d(ax4, body3d_1, type_str='hand', valid_idx=body_valid_1, color=np.array([1.0, 0.0, 0.0]))
        utils.general.plot3d(ax4, body3d_2, type_str='hand', valid_idx=body_valid_2, color=np.array([1.0, 0.0, 0.0]))
        ax4.set_xlabel('X Label')
        ax4.set_ylabel('Y Label')
        ax4.set_zlabel('Z Label')
        plt.axis('equal')

        xy, z = plot_all_PAF(PAF_1, 3)
        ax5 = fig.add_subplot(245)
        ax5.imshow(xy)

        ax6 = fig.add_subplot(246)
        ax6.imshow(z)

        xy, z = plot_all_PAF(PAF_2, 3)
        ax7 = fig.add_subplot(247)
        ax7.imshow(xy)

        ax8 = fig.add_subplot(248)
        ax8.imshow(z)

        plt.show()
