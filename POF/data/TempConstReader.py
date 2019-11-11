import tensorflow as tf
from data.BaseReader import BaseReader
import numpy as np


class TempConstReader(BaseReader):
    crop_scale_noise_sigma = 0.1
    crop_offset_noise_sigma = 0.1

    def __init__(self, objtype=0, shuffle=False, batch_size=1, crop_noise=False):
        super(TempConstReader, self).__init__(objtype, shuffle, batch_size, crop_noise)
        assert objtype in (0, 1), "This data reader only support single body / hands"

    def get(self, withPAF=True, read_image=True, imw=1920, imh=1080):
        # input to this data reader should have two consecutive frames
        # produce data from slice_input_producer
        flow_list = tf.train.slice_input_producer(list(self.tensor_dict.values()), shuffle=self.shuffle)
        flow_dict = {key: flow_list[ik] for ik, key in enumerate(self.tensor_dict.keys())}

        # build data dictionary
        data_dict = {}
        data_dict['1_img_dir'] = flow_dict['1_img_dirs']
        data_dict['2_img_dir'] = flow_dict['2_img_dirs']
        data_dict['1_K'] = flow_dict['1_K']
        data_dict['1_K'] = flow_dict['2_K']

        # rotate and project to camera frame
        if self.objtype == 0:
            body2d_1, body3d_1 = self.project_tf(flow_dict['1_body'], flow_dict['1_K'], flow_dict['1_R'], flow_dict['1_t'], flow_dict['1_distCoef'])
            body2d_2, body3d_2 = self.project_tf(flow_dict['2_body'], flow_dict['2_K'], flow_dict['2_R'], flow_dict['2_t'], flow_dict['2_distCoef'])
            body3d_1 = tf.cast(body3d_1, tf.float32)
            body3d_2 = tf.cast(body3d_2, tf.float32)
            body2d_1 = tf.cast(body2d_1, tf.float32)
            body2d_2 = tf.cast(body2d_2, tf.float32)
            data_dict['1_keypoint_xyz_origin'] = body3d_1
            data_dict['2_keypoint_xyz_origin'] = body3d_2
            data_dict['1_keypoint_uv_origin'] = body2d_1
            data_dict['2_keypoint_uv_origin'] = body2d_2
            data_dict['1_body_valid'] = flow_dict['1_body_valid']
            data_dict['2_body_valid'] = flow_dict['2_body_valid']
        elif self.objtype == 1:
            cond_left = tf.reduce_any(tf.cast(flow_dict['left_hand_valid'], dtype=tf.bool))  # 0 for right hand, 1 for left hand
            hand3d_1 = tf.cond(cond_left, lambda: flow_dict['1_left_hand'], lambda: flow_dict['1_right_hand'])  # in world coordinate
            hand3d_2 = tf.cond(cond_left, lambda: flow_dict['2_left_hand'], lambda: flow_dict['2_right_hand'])  # in world coordinate
            hand2d_1, hand3d_1 = self.project_tf(hand3d_1, flow_dict['1_K'], flow_dict['1_R'], flow_dict['1_t'], flow_dict['1_distCoef'])  # in camera coordinate
            hand2d_2, hand3d_2 = self.project_tf(hand3d_2, flow_dict['2_K'], flow_dict['2_R'], flow_dict['2_t'], flow_dict['2_distCoef'])  # in camera coordinate
            hand3d_1 = tf.cast(hand3d_1, tf.float32)
            hand3d_2 = tf.cast(hand3d_2, tf.float32)
            hand2d_1 = tf.cast(hand2d_1, tf.float32)
            hand2d_2 = tf.cast(hand2d_2, tf.float32)
            data_dict['1_keypoint_xyz_origin'] = hand3d_1
            data_dict['2_keypoint_xyz_origin'] = hand3d_2
            data_dict['1_keypoint_uv_origin'] = hand2d_1
            data_dict['2_keypoint_uv_origin'] = hand2d_2
            data_dict['cond_left'] = cond_left
            data_dict['left_hand_valid'] = flow_dict['left_hand_valid']
            data_dict['right_hand_valid'] = flow_dict['right_hand_valid']

        # read image
        if read_image:
            img_file_1 = tf.read_file(flow_dict['1_img_dirs'])
            img_file_2 = tf.read_file(flow_dict['2_img_dirs'])
            image_1 = tf.image.decode_image(img_file_1, channels=3)
            image_2 = tf.image.decode_image(img_file_2, channels=3)
            image_1 = tf.image.pad_to_bounding_box(image_1, 0, 0, imh, imw)
            image_2 = tf.image.pad_to_bounding_box(image_2, 0, 0, imh, imw)
            image_1.set_shape((imh, imw, 3))
            image_2.set_shape((imh, imw, 3))
            image_1 = tf.cast(image_1, tf.float32) / 255.0 - 0.5
            image_2 = tf.cast(image_2, tf.float32) / 255.0 - 0.5
            data_dict['1_image'] = image_1
            data_dict['2_image'] = image_2
        if 'mask_dirs_1' in flow_dict:
            assert 'mask_dirs_2' in flow_dict
            mask_file_1 = tf.read_file(flow_dict['1_mask_dirs'])
            mask_file_2 = tf.read_file(flow_dict['2_mask_dirs'])
            mask_1 = tf.image.decode_image(mask_file_1, channels=3)
            mask_2 = tf.image.decode_image(mask_file_2, channels=3)
            mask_1 = tf.image.pad_to_bounding_box(mask_1, 0, 0, imh, imw)
            mask_2 = tf.image.pad_to_bounding_box(mask_2, 0, 0, imh, imw)
            mask_1.set_shape((imh, imw, 3))
            mask_2.set_shape((imh, imw, 3))
            mask_1 = mask_1[:, :, 0]
            mask_2 = mask_2[:, :, 0]
            mask_1 = tf.cast(mask_1, tf.float32)
            mask_2 = tf.cast(mask_2, tf.float32)
        else:
            mask_1 = tf.ones((imh, imw), dtype=tf.float32)
            mask_2 = tf.ones((imh, imw), dtype=tf.float32)
        data_dict['1_mask'] = mask_1
        data_dict['2_mask'] = mask_2

        # calculate crop size
        if self.objtype in (0, 1):
            if self.objtype == 0:
                keypoints_1 = body3d_1
                keypoints_2 = body3d_2
                valid_1 = flow_dict['1_body_valid']
                valid_2 = flow_dict['2_body_valid']
            elif self.objtype == 1:
                keypoints_1 = hand3d_1
                keypoints_2 = hand3d_2
                valid_1 = tf.cond(cond_left, lambda: flow_dict['left_hand_valid'], lambda: flow_dict['right_hand_valid'])
                valid_2 = tf.cond(cond_left, lambda: flow_dict['left_hand_valid'], lambda: flow_dict['right_hand_valid'])
                data_dict['1_hand_valid'] = valid_1
                data_dict['2_hand_valid'] = valid_2
            crop_center3d_1, scale3d_1, crop_center2d_1, scale2d_1, crop_center3d_2, scale3d_2, crop_center2d_2, scale2d_2 = \
                self.calc_crop_scale_temp_const(keypoints_1, flow_dict['1_K'], flow_dict['1_distCoef'], valid_1, keypoints_2, flow_dict['2_K'], flow_dict['2_distCoef'], valid_2)
            data_dict['1_crop_center2d'], data_dict['1_scale2d'] = crop_center2d_1, scale2d_1
            data_dict['2_crop_center2d'], data_dict['2_scale2d'] = crop_center2d_2, scale2d_2
            data_dict['1_crop_center3d'], data_dict['1_scale3d'] = crop_center3d_1, scale3d_1
            data_dict['2_crop_center3d'], data_dict['2_scale3d'] = crop_center3d_2, scale3d_2

            # do cropping
            if self.objtype == 1:
                body2d_1 = hand2d_1
                body2d_2 = hand2d_2
                body3d_1 = hand3d_1
                body3d_2 = hand3d_2
            if self.rotate_augmentation:
                print('using rotation augmentation')
                rotate_angle_1 = tf.random_uniform([], minval=-np.pi * 40 / 180, maxval=np.pi * 40 / 180)
            else:
                rotate_angle_1 = 0.0
            rotate_angle_2 = tf.random_uniform([], minval=-np.pi * 5 / 180, maxval=np.pi * 5 / 180) + rotate_angle_1
            R2_1 = tf.reshape(tf.stack([tf.cos(rotate_angle_1), -tf.sin(rotate_angle_1), tf.sin(rotate_angle_1), tf.cos(rotate_angle_1)]), [2, 2])
            R2_2 = tf.reshape(tf.stack([tf.cos(rotate_angle_2), -tf.sin(rotate_angle_2), tf.sin(rotate_angle_2), tf.cos(rotate_angle_2)]), [2, 2])
            R3_1 = tf.reshape(tf.stack([tf.cos(rotate_angle_1), -tf.sin(rotate_angle_1), 0, tf.sin(rotate_angle_1), tf.cos(rotate_angle_1), 0, 0, 0, 1]), [3, 3])
            R3_2 = tf.reshape(tf.stack([tf.cos(rotate_angle_2), -tf.sin(rotate_angle_2), 0, tf.sin(rotate_angle_2), tf.cos(rotate_angle_2), 0, 0, 0, 1]), [3, 3])
            body2d_1 = tf.matmul((body2d_1 - crop_center2d_1), R2_1) + crop_center2d_1
            body2d_2 = tf.matmul((body2d_2 - crop_center2d_2), R2_2) + crop_center2d_2
            body3d_1 = tf.matmul((body3d_1 - crop_center3d_1), R3_1) + crop_center3d_1
            body3d_2 = tf.matmul((body3d_2 - crop_center3d_2), R3_2) + crop_center3d_2
            data_dict['1_keypoint_xyz_origin'] = body3d_1  # note that the projection of 3D might not be aligned with 2D any more after rotation
            data_dict['2_keypoint_xyz_origin'] = body3d_2  # note that the projection of 3D might not be aligned with 2D any more after rotation
            data_dict['1_keypoint_uv_origin'] = body2d_1
            data_dict['2_keypoint_uv_origin'] = body2d_2
            body2d_local_1 = self.update_keypoint2d(body2d_1, crop_center2d_1, scale2d_1)
            body2d_local_2 = self.update_keypoint2d(body2d_2, crop_center2d_2, scale2d_2)
            data_dict['1_keypoint_uv_local'] = body2d_local_1
            data_dict['2_keypoint_uv_local'] = body2d_local_2

            if read_image:
                image_crop_1 = self.crop_image(image_1, crop_center2d_1, scale2d_1)
                image_crop_2 = self.crop_image(image_2, crop_center2d_2, scale2d_2)
                data_dict['1_image_crop'] = image_crop_1
                data_dict['2_image_crop'] = image_crop_2
            mask_crop_1 = self.crop_image(tf.stack([mask_1] * 3, axis=2), crop_center2d_1, scale2d_1)
            mask_crop_2 = self.crop_image(tf.stack([mask_2] * 3, axis=2), crop_center2d_2, scale2d_2)
            data_dict['1_mask_crop'] = mask_crop_1[:, :, 0]
            data_dict['2_mask_crop'] = mask_crop_2[:, :, 0]

            data_dict['1_image_crop'] = tf.contrib.image.rotate(data_dict['1_image_crop'], rotate_angle_1)
            data_dict['2_image_crop'] = tf.contrib.image.rotate(data_dict['2_image_crop'], rotate_angle_2)
            data_dict['1_mask_crop'] = tf.contrib.image.rotate(data_dict['1_mask_crop'], rotate_angle_1)
            data_dict['2_mask_crop'] = tf.contrib.image.rotate(data_dict['2_mask_crop'], rotate_angle_2)
            if self.blur_augmentation:
                print('using blur augmentation')
                rescale_factor = tf.random_uniform([], minval=0.1, maxval=1.0)
                rescale = tf.cast(rescale_factor * self.crop_size, tf.int32)
                resized_image_1 = tf.image.resize_images(data_dict['1_image_crop'], [rescale, rescale])
                resized_image_2 = tf.image.resize_images(data_dict['2_image_crop'], [rescale, rescale])
                data_dict['1_image_crop'] = tf.image.resize_images(resized_image_1, [self.crop_size, self.crop_size])
                data_dict['2_image_crop'] = tf.image.resize_images(resized_image_2, [self.crop_size, self.crop_size])

            # create 2D gaussian map
            scoremap2d_1 = self.create_multiple_gaussian_map(body2d_local_1[:, ::-1], (self.crop_size, self.crop_size), self.sigma, valid_vec=valid_1, extra=True)  # coord_hw, imsize_hw
            scoremap2d_2 = self.create_multiple_gaussian_map(body2d_local_2[:, ::-1], (self.crop_size, self.crop_size), self.sigma, valid_vec=valid_2, extra=True)  # coord_hw, imsize_hw
            data_dict['1_scoremap2d'] = scoremap2d_1
            data_dict['2_scoremap2d'] = scoremap2d_2

            if withPAF:
                from utils.PAF import createPAF
                data_dict['1_PAF'] = createPAF(body2d_local_1, body3d_1, self.objtype, (self.crop_size, self.crop_size), True, valid_vec=valid_1)
                data_dict['2_PAF'] = createPAF(body2d_local_2, body3d_2, self.objtype, (self.crop_size, self.crop_size), True, valid_vec=valid_2)
                data_dict['1_PAF_type'] = tf.ones([], dtype=bool)  # 0 for 2D PAF, 1 for 3D PAF
                data_dict['2_PAF_type'] = tf.ones([], dtype=bool)  # 0 for 2D PAF, 1 for 3D PAF

            # create 3D gaussian_map
            body3d_local_1 = self.update_keypoint3d(body3d_1, crop_center3d_1, scale3d_1)
            body3d_local_2 = self.update_keypoint3d(body3d_2, crop_center3d_2, scale3d_2)
            data_dict['1_keypoint_xyz_local'] = body3d_local_1
            data_dict['2_keypoint_xyz_local'] = body3d_local_2
            # scoremap3d = self.create_multiple_gaussian_map_3d(body3d_local, self.grid_size, self.sigma3d, valid_vec=valid, extra=True)
            # data_dict['1_scoremap3d'] = scoremap3d

            if self.objtype == 1:  # this is hand, flip the image if it is right hand
                data_dict['1_image_crop'] = tf.cond(cond_left, lambda: data_dict['1_image_crop'], lambda: data_dict['1_image_crop'][:, ::-1, :])
                data_dict['2_image_crop'] = tf.cond(cond_left, lambda: data_dict['2_image_crop'], lambda: data_dict['2_image_crop'][:, ::-1, :])
                data_dict['1_mask_crop'] = tf.cond(cond_left, lambda: data_dict['1_mask_crop'], lambda: data_dict['1_mask_crop'][:, ::-1])
                data_dict['2_mask_crop'] = tf.cond(cond_left, lambda: data_dict['2_mask_crop'], lambda: data_dict['2_mask_crop'][:, ::-1])
                data_dict['1_scoremap2d'] = tf.cond(cond_left, lambda: data_dict['1_scoremap2d'], lambda: data_dict['1_scoremap2d'][:, ::-1, :])
                data_dict['2_scoremap2d'] = tf.cond(cond_left, lambda: data_dict['2_scoremap2d'], lambda: data_dict['2_scoremap2d'][:, ::-1, :])
                data_dict['1_keypoint_uv_local'] = tf.cond(cond_left, lambda: data_dict['1_keypoint_uv_local'],
                                                           lambda: tf.constant([self.crop_size, 0], tf.float32) + tf.constant([-1, 1], tf.float32) * data_dict['1_keypoint_uv_local'])
                data_dict['2_keypoint_uv_local'] = tf.cond(cond_left, lambda: data_dict['2_keypoint_uv_local'],
                                                           lambda: tf.constant([self.crop_size, 0], tf.float32) + tf.constant([-1, 1], tf.float32) * data_dict['2_keypoint_uv_local'])
                if withPAF:
                    data_dict['1_PAF'] = tf.cond(cond_left, lambda: data_dict['1_PAF'],
                                                 lambda: (data_dict['1_PAF'][:, ::-1, :]) * tf.constant([-1, 1, 1] * (data_dict['1_PAF'].get_shape().as_list()[2] // 3), dtype=tf.float32))
                    data_dict['2_PAF'] = tf.cond(cond_left, lambda: data_dict['2_PAF'],
                                                 lambda: (data_dict['2_PAF'][:, ::-1, :]) * tf.constant([-1, 1, 1] * (data_dict['2_PAF'].get_shape().as_list()[2] // 3), dtype=tf.float32))

        names, tensors = zip(*data_dict.items())

        if self.shuffle:
            tensors = tf.train.shuffle_batch_join([tensors],
                                                  batch_size=self.batch_size,
                                                  capacity=100,
                                                  min_after_dequeue=20,
                                                  enqueue_many=False)
        else:
            tensors = tf.train.batch_join([tensors],
                                          batch_size=self.batch_size,
                                          capacity=20,
                                          enqueue_many=False)

        return dict(zip(names, tensors))

    def calc_crop_scale_temp_const(self, keypoints_1, calibK_1, calibDC_1, valid_1, keypoints_2, calibK_2, calibDC_2, valid_2):
        if self.objtype == 0:
            keypoint_center_1 = (keypoints_1[8] + keypoints_1[11]) / 2
            keypoint_center_2 = (keypoints_2[8] + keypoints_2[11]) / 2
            center_valid_1 = tf.logical_and(valid_1[8], valid_1[11])
            center_valid_2 = tf.logical_and(valid_2[8], valid_2[11])
        elif self.objtype == 1:
            keypoint_center_1 = keypoints_1[12]
            keypoint_center_2 = keypoints_2[12]
            center_valid_1 = valid_1[12]
            center_valid_2 = valid_2[12]
        else:
            raise NotImplementedError

        valid_idx_1 = tf.where(valid_1)[:, 0]
        valid_idx_2 = tf.where(valid_2)[:, 0]
        valid_keypoints_1 = tf.gather(keypoints_1, valid_idx_1, name='1_valid_keypoints')
        valid_keypoints_2 = tf.gather(keypoints_2, valid_idx_2, name='2_valid_keypoints')

        min_coord_1 = tf.reduce_min(valid_keypoints_1, 0, name='1_min_coord')
        min_coord_2 = tf.reduce_min(valid_keypoints_2, 0, name='2_min_coord')
        max_coord_1 = tf.reduce_max(valid_keypoints_1, 0, name='1_max_coord')
        max_coord_2 = tf.reduce_max(valid_keypoints_2, 0, name='2_max_coord')

        keypoint_center_1 = tf.cond(center_valid_1, lambda: keypoint_center_1, lambda: (min_coord_1 + max_coord_1) / 2)
        keypoint_center_2 = tf.cond(center_valid_2, lambda: keypoint_center_2, lambda: (min_coord_2 + max_coord_2) / 2)
        keypoint_center_1.set_shape((3,))
        keypoint_center_2.set_shape((3,))

        fit_size_1 = tf.reduce_max(tf.maximum(max_coord_1 - keypoint_center_1, keypoint_center_1 - min_coord_1))
        fit_size_2 = tf.reduce_max(tf.maximum(max_coord_2 - keypoint_center_2, keypoint_center_2 - min_coord_2))
        crop_scale_noise_1 = tf.cast(1.0, tf.float32)
        if self.crop_noise:
            crop_scale_noise_1 = tf.exp(tf.truncated_normal([], mean=0.0, stddev=self.crop_scale_noise_sigma))
            crop_scale_noise_1 = tf.maximum(crop_scale_noise_1, tf.reciprocal(self.crop_size_zoom))
        crop_scale_noise_2 = crop_scale_noise_1 + tf.truncated_normal([], mean=0.0, stddev=0.01)
        crop_size_best_1 = tf.multiply(crop_scale_noise_1, 2 * fit_size_1 * self.crop_size_zoom, name='1_crop_size_best')
        crop_size_best_2 = tf.multiply(crop_scale_noise_2, 2 * fit_size_2 * self.crop_size_zoom, name='2_crop_size_best')

        crop_offset_noise_1 = tf.cast(0.0, tf.float32)
        if self.crop_noise:
            crop_offset_noise_1 = tf.truncated_normal([3], mean=0.0, stddev=self.crop_offset_noise_sigma) * fit_size_1 * tf.constant([1., 1., 0.], dtype=tf.float32)
        crop_offset_noise_2 = tf.truncated_normal([3], mean=0.0, stddev=0.01) * fit_size_2 * tf.constant([1., 1., 0.], dtype=tf.float32) + crop_offset_noise_1
        crop_offset_noise_1 = tf.maximum(crop_offset_noise_1, max_coord_1 + 1e-5 - crop_size_best_1 / 2 - keypoint_center_1)
        crop_offset_noise_2 = tf.maximum(crop_offset_noise_2, max_coord_2 + 1e-5 - crop_size_best_2 / 2 - keypoint_center_2)
        crop_offset_noise_1 = tf.minimum(crop_offset_noise_1, min_coord_1 - 1e-5 + crop_size_best_1 / 2 - keypoint_center_1, name='1_crop_offset_noise')
        crop_offset_noise_2 = tf.minimum(crop_offset_noise_2, min_coord_2 - 1e-5 + crop_size_best_2 / 2 - keypoint_center_2, name='2_crop_offset_noise')
        crop_center_1 = tf.add(keypoint_center_1, crop_offset_noise_1, name='1_crop_center')
        crop_center_2 = tf.add(keypoint_center_2, crop_offset_noise_2, name='2_crop_center')

        crop_box_bl_1 = tf.concat([crop_center_1[:2] - crop_size_best_1 / 2, crop_center_1[2:]], 0)
        crop_box_bl_2 = tf.concat([crop_center_2[:2] - crop_size_best_2 / 2, crop_center_2[2:]], 0)
        crop_box_ur_1 = tf.concat([crop_center_1[:2] + crop_size_best_1 / 2, crop_center_1[2:]], 0)
        crop_box_ur_2 = tf.concat([crop_center_2[:2] + crop_size_best_2 / 2, crop_center_2[2:]], 0)

        crop_box_1 = tf.stack([crop_box_bl_1, crop_box_ur_1], 0)
        crop_box_2 = tf.stack([crop_box_bl_2, crop_box_ur_2], 0)
        scale_1 = tf.cast(self.grid_size, tf.float32) / crop_size_best_1
        scale_2 = tf.cast(self.grid_size, tf.float32) / crop_size_best_2

        crop_box2d_1, _ = self.project_tf(crop_box_1, calibK_1, calibDistCoef=calibDC_1)
        crop_box2d_2, _ = self.project_tf(crop_box_2, calibK_2, calibDistCoef=calibDC_2)
        min_coord2d_1 = tf.reduce_min(crop_box2d_1, 0)
        min_coord2d_2 = tf.reduce_min(crop_box2d_2, 0)
        max_coord2d_1 = tf.reduce_max(crop_box2d_1, 0)
        max_coord2d_2 = tf.reduce_max(crop_box2d_2, 0)
        crop_size_best2d_1 = tf.reduce_max(max_coord2d_1 - min_coord2d_1)
        crop_size_best2d_2 = tf.reduce_max(max_coord2d_2 - min_coord2d_2)
        crop_center2d_1 = (min_coord2d_1 + max_coord2d_1) / 2
        crop_center2d_2 = (min_coord2d_2 + max_coord2d_2) / 2
        scale2d_1 = tf.cast(self.crop_size, tf.float32) / crop_size_best2d_1
        scale2d_2 = tf.cast(self.crop_size, tf.float32) / crop_size_best2d_2
        return crop_center_1, scale_1, crop_center2d_1, scale2d_1, crop_center_2, scale_2, crop_center2d_2, scale2d_2

    @staticmethod
    def convertToSingleFrameDataWithPrevGT(data_dict):
        out_dict = {}
        out_dict['scoremap2d'] = data_dict['2_scoremap2d']
        if '2_hand_valid' in data_dict:
            out_dict['hand_valid'] = data_dict['2_hand_valid']
        elif '2_body_valid' in data_dict:
            out_dict['body_valid'] = data_dict['2_body_valid']
        out_dict['PAF'] = data_dict['2_PAF']
        out_dict['PAF_type'] = data_dict['2_PAF_type']
        out_dict['mask_crop'] = data_dict['2_mask_crop']
        out_dict['image_crop'] = tf.concat([data_dict['2_image_crop'], data_dict['1_image_crop'], data_dict['1_scoremap2d'], data_dict['1_PAF']], axis=3)
        return out_dict

    @staticmethod
    def convertToSingleFrameDataWithPrevOutput(data_dict):
        out_dict = {}
        out_dict['scoremap2d'] = data_dict['2_scoremap2d']
        if '2_hand_valid' in data_dict:
            out_dict['hand_valid'] = data_dict['2_hand_valid']
        elif '2_body_valid' in data_dict:
            out_dict['body_valid'] = data_dict['2_body_valid']
        out_dict['PAF'] = data_dict['2_PAF']
        out_dict['PAF_type'] = data_dict['2_PAF_type']
        out_dict['mask_crop'] = data_dict['2_mask_crop']
        out_dict['image_crop'] = tf.concat([data_dict['2_image_crop'], data_dict['1_image_crop']], axis=3)
        out_dict['pre_input'] = data_dict['pre_input']
        out_dict['temp_data'] = data_dict['temp_data']
        return out_dict
