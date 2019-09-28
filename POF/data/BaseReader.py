import tensorflow as tf
import numpy as np
import utils.general


class BaseReader(object):
    # BaseReader is a virual base class to be inherited by other data readers which provide data by calling register_tensor

    crop_size_zoom = 1.5
    crop_size_zoom_2d = 1.8
    crop_size = 368
    grid_size = crop_size // 8
    sigma = 7
    sigma3d = 3
    rotate_augmentation = False
    blur_augmentation = False
    crop_scale_noise_sigma = 0.1
    crop_offset_noise_sigma = 0.1
    crop_scale_noise_sigma_2d = 0.1
    crop_offset_noise_sigma_2d = 0.1

    def __init__(self, objtype=0, shuffle=True, batch_size=1, crop_noise=False):
        # objtype: 0 = body only, 1 = hand only, 2 = body and hands
        assert objtype in (0, 1, 2)
        self.objtype = objtype
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.crop_noise = crop_noise

    def register_tensor(self, data_dict, order_dict):
        l = [len(value) for value in data_dict.values() if len(value) > 0]
        assert len(set(l)) == 1  # check the length of all data items to be consistent
        print('loading dataset of size {}'.format(l[0]))
        self.tensor_dict = {}
        for key, value in data_dict.items():
            if len(value) > 0:
                value = np.array(value)
                if key in order_dict:
                    self.tensor_dict[key] = self.switch_joint_order(value, order_dict[key])
                else:
                    self.tensor_dict[key] = value

    def get(self, withPAF=True, PAF_normalize3d=True, read_image=True, imw=1920, imh=1080, bbox2d=0):
        # bbox2d: 0: computed from 3D bounding box, 1: compute from openpose
        assert bbox2d in (0, 1)
        assert type(withPAF) == bool

        # produce data from slice_input_producer
        flow_list = tf.train.slice_input_producer(list(self.tensor_dict.values()), shuffle=self.shuffle)
        flow_dict = {key: flow_list[ik] for ik, key in enumerate(self.tensor_dict.keys())}

        # build data dictionary
        data_dict = {}
        data_dict['img_dir'] = flow_dict['img_dirs']
        data_dict['K'] = flow_dict['K']

        # rotate and project to camera frame
        if self.objtype == 0:
            body2d, body3d = self.project_tf(flow_dict['body'], flow_dict['K'], flow_dict['R'], flow_dict['t'], flow_dict['distCoef'])
            body3d = tf.cast(body3d, tf.float32)
            body2d = tf.cast(body2d, tf.float32)
            data_dict['keypoint_xyz_origin'] = body3d
            data_dict['keypoint_uv_origin'] = body2d
            data_dict['body_valid'] = flow_dict['body_valid']
        elif self.objtype == 1:
            cond_left = tf.reduce_any(tf.cast(flow_dict['left_hand_valid'], dtype=tf.bool))  # 0 for right hand, 1 for left hand
            hand3d = tf.cond(cond_left, lambda: flow_dict['left_hand'], lambda: flow_dict['right_hand'])  # in world coordinate
            hand2d, hand3d = self.project_tf(hand3d, flow_dict['K'], flow_dict['R'], flow_dict['t'], flow_dict['distCoef'])  # in camera coordinate
            hand3d = tf.cast(hand3d, tf.float32)
            hand2d = tf.cast(hand2d, tf.float32)
            data_dict['keypoint_xyz_origin'] = hand3d
            data_dict['keypoint_uv_origin'] = hand2d
            data_dict['cond_left'] = cond_left
            data_dict['left_hand_valid'] = flow_dict['left_hand_valid']
            data_dict['right_hand_valid'] = flow_dict['right_hand_valid']
        elif self.objtype == 2:
            body2d, body3d = self.project_tf(flow_dict['body'], flow_dict['K'], flow_dict['R'], flow_dict['t'], flow_dict['distCoef'])
            lhand2d, lhand3d = self.project_tf(flow_dict['left_hand'], flow_dict['K'], flow_dict['R'], flow_dict['t'], flow_dict['distCoef'])
            rhand2d, rhand3d = self.project_tf(flow_dict['right_hand'], flow_dict['K'], flow_dict['R'], flow_dict['t'], flow_dict['distCoef'])
            data_dict['body_xyz_origin'] = body3d
            data_dict['body_uv_origin'] = body2d
            data_dict['lhand_xyz_origin'] = lhand3d
            data_dict['lhand_uv_origin'] = lhand2d
            data_dict['rhand_xyz_origin'] = rhand3d
            data_dict['rhand_uv_origin'] = rhand2d
            data_dict['body_valid'] = flow_dict['body_valid']
            data_dict['left_hand_valid'] = flow_dict['left_hand_valid']
            data_dict['right_hand_valid'] = flow_dict['right_hand_valid']

        # read image
        if read_image:
            img_file = tf.read_file(flow_dict['img_dirs'])
            image = tf.image.decode_image(img_file, channels=3)
            image = tf.image.pad_to_bounding_box(image, 0, 0, imh, imw)
            image.set_shape((imh, imw, 3))
            image = tf.cast(image, tf.float32) / 255.0 - 0.5
            data_dict['image'] = image
        if 'mask_dirs' in flow_dict:
            mask_file = tf.read_file(flow_dict['mask_dirs'])
            mask = tf.image.decode_image(mask_file, channels=3)
            mask = tf.image.pad_to_bounding_box(mask, 0, 0, imh, imw)
            mask.set_shape((imh, imw, 3))
            mask = mask[:, :, 0]
            mask = tf.cast(mask, tf.float32)
        else:
            mask = tf.ones((imh, imw), dtype=tf.float32)
        data_dict['mask'] = mask

        # calculate crop size
        if self.objtype in (0, 1):
            if self.objtype == 0:
                keypoints = body3d
                valid = flow_dict['body_valid']
            elif self.objtype == 1:
                keypoints = hand3d
                valid = tf.cond(cond_left, lambda: flow_dict['left_hand_valid'], lambda: flow_dict['right_hand_valid'])
                data_dict['hand_valid'] = valid
            crop_center3d, scale3d, crop_center2d, scale2d = self.calc_crop_scale(keypoints, flow_dict['K'], flow_dict['distCoef'], valid)
            data_dict['crop_center2d'], data_dict['scale2d'] = crop_center2d, scale2d
            data_dict['crop_center3d'], data_dict['scale3d'] = crop_center3d, scale3d

            # do cropping
            if self.objtype == 1:
                body2d = hand2d
                body3d = hand3d
            if self.rotate_augmentation:
                print('using rotation augmentation')
                rotate_angle = tf.random_uniform([], minval=-np.pi * 40 / 180, maxval=np.pi * 40 / 180)
                R2 = tf.reshape(tf.stack([tf.cos(rotate_angle), -tf.sin(rotate_angle), tf.sin(rotate_angle), tf.cos(rotate_angle)]), [2, 2])
                R3 = tf.reshape(tf.stack([tf.cos(rotate_angle), -tf.sin(rotate_angle), 0, tf.sin(rotate_angle), tf.cos(rotate_angle), 0, 0, 0, 1]), [3, 3])
                body2d = tf.matmul((body2d - crop_center2d), R2) + crop_center2d
                body3d = tf.matmul((body3d - crop_center3d), R3) + crop_center3d
                data_dict['keypoint_xyz_origin'] = body3d  # note that the projection of 3D might not be aligned with 2D any more after rotation
                data_dict['keypoint_uv_origin'] = body2d
            body2d_local = self.update_keypoint2d(body2d, crop_center2d, scale2d)
            data_dict['keypoint_uv_local'] = body2d_local
            if read_image:
                image_crop = self.crop_image(image, crop_center2d, scale2d)
                data_dict['image_crop'] = image_crop
            mask_crop = self.crop_image(tf.stack([mask] * 3, axis=2), crop_center2d, scale2d)
            data_dict['mask_crop'] = mask_crop[:, :, 0]
            if self.rotate_augmentation:
                data_dict['image_crop'] = tf.contrib.image.rotate(data_dict['image_crop'], rotate_angle)
                data_dict['mask_crop'] = tf.contrib.image.rotate(data_dict['mask_crop'], rotate_angle)
            if self.blur_augmentation:
                print('using blur augmentation')
                rescale_factor = tf.random_uniform([], minval=0.1, maxval=1.0)
                rescale = tf.cast(rescale_factor * self.crop_size, tf.int32)
                resized_image = tf.image.resize_images(data_dict['image_crop'], [rescale, rescale])
                data_dict['image_crop'] = tf.image.resize_images(resized_image, [self.crop_size, self.crop_size])

            # create 2D gaussian map
            scoremap2d = self.create_multiple_gaussian_map(body2d_local[:, ::-1], (self.crop_size, self.crop_size), self.sigma, valid_vec=valid, extra=True)  # coord_hw, imsize_hw
            data_dict['scoremap2d'] = scoremap2d

            if withPAF:
                from utils.PAF import createPAF
                data_dict['PAF'] = createPAF(body2d_local, body3d, self.objtype, (self.crop_size, self.crop_size), PAF_normalize3d, valid_vec=valid)
                data_dict['PAF_type'] = tf.ones([], dtype=bool)  # 0 for 2D PAF, 1 for 3D PAF

            # create 3D gaussian_map
            body3d_local = self.update_keypoint3d(body3d, crop_center3d, scale3d)
            data_dict['keypoint_xyz_local'] = body3d_local
            # scoremap3d = self.create_multiple_gaussian_map_3d(body3d_local, self.grid_size, self.sigma3d, valid_vec=valid, extra=True)
            # data_dict['scoremap3d'] = scoremap3d

            if self.objtype == 1:  # this is hand, flip the image if it is right hand
                data_dict['image_crop'] = tf.cond(cond_left, lambda: data_dict['image_crop'], lambda: data_dict['image_crop'][:, ::-1, :])
                data_dict['mask_crop'] = tf.cond(cond_left, lambda: data_dict['mask_crop'], lambda: data_dict['mask_crop'][:, ::-1])
                data_dict['scoremap2d'] = tf.cond(cond_left, lambda: data_dict['scoremap2d'], lambda: data_dict['scoremap2d'][:, ::-1, :])
                data_dict['keypoint_uv_local'] = tf.cond(cond_left, lambda: data_dict['keypoint_uv_local'],
                                                         lambda: tf.constant([self.crop_size, 0], tf.float32) + tf.constant([-1, 1], tf.float32) * data_dict['keypoint_uv_local'])
                if withPAF:
                    data_dict['PAF'] = tf.cond(cond_left, lambda: data_dict['PAF'],
                                               lambda: (data_dict['PAF'][:, ::-1, :]) * tf.constant([-1, 1, 1] * (data_dict['PAF'].get_shape().as_list()[2] // 3), dtype=tf.float32))

        elif self.objtype == 2:
            bcrop_center3d, bscale3d, bcrop_center2d, bscale2d = self.calc_crop_scale(body3d, flow_dict['K'], flow_dict['distCoef'], flow_dict['body_valid'])
            lcrop_center3d, lscale3d, lcrop_center2d, lscale2d = self.calc_crop_scale(lhand3d, flow_dict['K'], flow_dict['distCoef'], flow_dict['left_hand_valid'])
            rcrop_center3d, rscale3d, rcrop_center2d, rscale2d = self.calc_crop_scale(rhand3d, flow_dict['K'], flow_dict['distCoef'], flow_dict['right_hand_valid'])

            body3d_local = self.update_keypoint3d(body3d, bcrop_center3d, bscale3d)
            lhand3d_local = self.update_keypoint3d(lhand3d, lcrop_center3d, lscale3d)
            rhand3d_local = self.update_keypoint3d(rhand3d, rcrop_center3d, rscale3d)
            bscoremap3d = self.create_multiple_gaussian_map_3d(body3d_local, self.grid_size, self.sigma3d,
                                                               valid_vec=flow_dict['body_valid'], extra=True)  # coord_hw, imsize_hw
            lscoremap3d = self.create_multiple_gaussian_map_3d(lhand3d_local, self.grid_size, self.sigma3d,
                                                               valid_vec=flow_dict['left_hand_valid'], extra=True)  # coord_hw, imsize_hw
            rscoremap3d = self.create_multiple_gaussian_map_3d(rhand3d_local, self.grid_size, self.sigma3d,
                                                               valid_vec=flow_dict['right_hand_valid'], extra=True)  # coord_hw, imsize_hw
            data_dict['bscoremap3d'] = bscoremap3d
            data_dict['lscoremap3d'] = lscoremap3d
            data_dict['rscoremap3d'] = rscoremap3d

            data_dict['body_xyz_local'] = body3d_local
            data_dict['lhand_xyz_local'] = lhand3d_local
            data_dict['rhand_xyz_local'] = rhand3d_local

            # 2D keypoints and cropped images
            if bbox2d == 1:
                # crop the 2D bounding box from openpose data
                body2d = flow_dict['openpose_body']
                lhand2d = flow_dict['openpose_lhand']
                rhand2d = flow_dict['openpose_rhand']

                bvalid = tf.logical_and(tf.not_equal(body2d[:, 0], 0.0), tf.not_equal(body2d[:, 1], 0.0))
                lvalid = tf.logical_and(tf.not_equal(lhand2d[:, 0], 0.0), tf.not_equal(lhand2d[:, 1], 0.0))
                rvalid = tf.logical_and(tf.not_equal(rhand2d[:, 0], 0.0), tf.not_equal(rhand2d[:, 1], 0.0))

                data_dict['body_valid'] = bvalid
                data_dict['left_hand_valid'] = lvalid
                data_dict['right_hand_valid'] = rvalid

                if 'openpose_foot' in flow_dict:
                    data_dict['openpose_foot'] = flow_dict['openpose_foot']

                bcrop_center2d, bscale2d = self.calc_crop_scale2d(body2d, bvalid)
                lcrop_center2d, lscale2d = self.calc_crop_scale2d(lhand2d, lvalid)
                rcrop_center2d, rscale2d = self.calc_crop_scale2d(rhand2d, rvalid)

            body2d_local = self.update_keypoint2d(body2d, bcrop_center2d, bscale2d)
            lhand2d_local = self.update_keypoint2d(lhand2d, lcrop_center2d, lscale2d)
            rhand2d_local = self.update_keypoint2d(rhand2d, rcrop_center2d, rscale2d)

            data_dict['body_uv_local'] = body2d_local
            data_dict['lhand_uv_local'] = lhand2d_local
            data_dict['rhand_uv_local'] = rhand2d_local
            data_dict['bcrop_center2d'] = bcrop_center2d
            data_dict['lcrop_center2d'] = lcrop_center2d
            data_dict['rcrop_center2d'] = rcrop_center2d
            data_dict['bscale2d'] = bscale2d
            data_dict['lscale2d'] = lscale2d
            data_dict['rscale2d'] = rscale2d

            if read_image:
                bimage_crop = self.crop_image(image, bcrop_center2d, bscale2d)
                limage_crop = self.crop_image(image, lcrop_center2d, lscale2d)
                rimage_crop = self.crop_image(image, rcrop_center2d, rscale2d)
                data_dict['bimage_crop'] = bimage_crop
                data_dict['limage_crop'] = limage_crop
                data_dict['rimage_crop'] = rimage_crop

            bscoremap2d = self.create_multiple_gaussian_map(body2d_local[:, ::-1], (self.crop_size, self.crop_size), self.sigma,
                                                            valid_vec=flow_dict['body_valid'], extra=True)  # coord_hw, imsize_hw
            lscoremap2d = self.create_multiple_gaussian_map(lhand2d_local[:, ::-1], (self.crop_size, self.crop_size), self.sigma,
                                                            valid_vec=flow_dict['left_hand_valid'], extra=True)  # coord_hw, imsize_hw
            rscoremap2d = self.create_multiple_gaussian_map(rhand2d_local[:, ::-1], (self.crop_size, self.crop_size), self.sigma,
                                                            valid_vec=flow_dict['right_hand_valid'], extra=True)  # coord_hw, imsize_hw
            data_dict['bscoremap2d'] = bscoremap2d
            data_dict['lscoremap2d'] = lscoremap2d
            data_dict['rscoremap2d'] = rscoremap2d

            # for openpose data
            for key, val in flow_dict.items():
                if 'openpose' not in key:
                    continue
                data_dict[key] = val

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

    def calc_crop_scale(self, keypoints, calibK, calibDC, valid):
        if self.objtype == 0:
            keypoint_center = (keypoints[8] + keypoints[11]) / 2
            center_valid = tf.logical_and(valid[8], valid[11])
        elif self.objtype == 1:
            keypoint_center = keypoints[12]
            center_valid = valid[12]
        else:  # objtype == 2
            assert self.objtype == 2  # conditioned by the shape of input
            if keypoints.shape[0] == 18:
                keypoint_center = (keypoints[8] + keypoints[11]) / 2
                center_valid = tf.logical_and(valid[8], valid[11])
            else:
                keypoint_center = keypoints[12]
                center_valid = valid[12]

        valid_idx = tf.where(valid)[:, 0]
        valid_keypoints = tf.gather(keypoints, valid_idx, name='valid_keypoints')

        min_coord = tf.reduce_min(valid_keypoints, 0, name='min_coord')
        max_coord = tf.reduce_max(valid_keypoints, 0, name='max_coord')

        keypoint_center = tf.cond(center_valid, lambda: keypoint_center, lambda: (min_coord + max_coord) / 2)
        keypoint_center.set_shape((3,))

        fit_size = tf.reduce_max(tf.maximum(max_coord - keypoint_center, keypoint_center - min_coord))
        crop_scale_noise = tf.cast(1.0, tf.float32)
        if self.crop_noise:
            crop_scale_noise = tf.exp(tf.truncated_normal([], mean=0.0, stddev=self.crop_scale_noise_sigma))
            crop_scale_noise = tf.maximum(crop_scale_noise, tf.reciprocal(self.crop_size_zoom))
        crop_size_best = tf.multiply(crop_scale_noise, 2 * fit_size * self.crop_size_zoom, name='crop_size_best')

        crop_offset_noise = tf.cast(0.0, tf.float32)
        if self.crop_noise:
            crop_offset_noise = tf.truncated_normal([3], mean=0.0, stddev=self.crop_offset_noise_sigma) * fit_size * tf.constant([1., 1., 0.], dtype=tf.float32)
            crop_offset_noise = tf.maximum(crop_offset_noise, max_coord + 1e-5 - crop_size_best / 2 - keypoint_center)
            crop_offset_noise = tf.minimum(crop_offset_noise, min_coord - 1e-5 + crop_size_best / 2 - keypoint_center, name='crop_offset_noise')
        crop_center = tf.add(keypoint_center, crop_offset_noise, name='crop_center')

        crop_box_bl = tf.concat([crop_center[:2] - crop_size_best / 2, crop_center[2:]], 0)
        crop_box_ur = tf.concat([crop_center[:2] + crop_size_best / 2, crop_center[2:]], 0)

        crop_box = tf.stack([crop_box_bl, crop_box_ur], 0)
        scale = tf.cast(self.grid_size, tf.float32) / crop_size_best

        crop_box2d, _ = self.project_tf(crop_box, calibK, calibDistCoef=calibDC)
        min_coord2d = tf.reduce_min(crop_box2d, 0)
        max_coord2d = tf.reduce_max(crop_box2d, 0)
        crop_size_best2d = tf.reduce_max(max_coord2d - min_coord2d)
        crop_center2d = (min_coord2d + max_coord2d) / 2
        scale2d = tf.cast(self.crop_size, tf.float32) / crop_size_best2d
        return crop_center, scale, crop_center2d, scale2d

    def calc_crop_scale2d(self, keypoints, valid):
        # assert self.objtype == 2
        if keypoints.shape[0] == 19 or keypoints.shape[0] == 20:
            keypoint_center = (keypoints[8] + keypoints[11]) / 2
            center_valid = tf.logical_and(valid[8], valid[11])
        else:
            keypoint_center = keypoints[12]
            center_valid = valid[12]

        valid_idx = tf.where(valid)[:, 0]
        valid_keypoints = tf.gather(keypoints, valid_idx)
        min_coord = tf.reduce_min(valid_keypoints, 0)
        max_coord = tf.reduce_max(valid_keypoints, 0)

        keypoint_center = tf.cond(center_valid, lambda: keypoint_center, lambda: (min_coord + max_coord) / 2)
        keypoint_center.set_shape((2,))

        fit_size = tf.reduce_max(tf.maximum(max_coord - keypoint_center, keypoint_center - min_coord))
        crop_scale_noise = tf.cast(1.0, tf.float32)
        if self.crop_noise:
            crop_scale_noise = tf.exp(tf.truncated_normal([], mean=0.0, stddev=self.crop_scale_noise_sigma_2d))
        crop_size_best = 2 * fit_size * self.crop_size_zoom_2d * crop_scale_noise

        crop_offset_noise = tf.cast(0.0, tf.float32)
        if self.crop_noise:
            crop_offset_noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma_2d) * fit_size
            crop_offset_noise = tf.maximum(crop_offset_noise, keypoint_center - crop_size_best / 2 - min_coord + 1)
            crop_offset_noise = tf.minimum(crop_offset_noise, keypoint_center + crop_size_best / 2 - max_coord - 1)
        crop_center = keypoint_center + crop_offset_noise
        scale2d = tf.cast(self.crop_size, tf.float32) / crop_size_best
        return crop_center, scale2d

    def crop_image(self, image, crop_center2d, scale2d):
        image_crop = utils.general.crop_image_from_xy(tf.expand_dims(image, 0), crop_center2d[::-1], self.crop_size, scale2d)  # crop_center_hw
        image_crop = tf.squeeze(image_crop)
        return image_crop

    def update_keypoint2d(self, keypoint2d, crop_center2d, scale2d):
        keypoint_x = (keypoint2d[:, 0] - crop_center2d[0]) * scale2d + self.crop_size // 2
        keypoint_y = (keypoint2d[:, 1] - crop_center2d[1]) * scale2d + self.crop_size // 2
        keypoint2d_local = tf.stack([keypoint_x, keypoint_y], 1)
        keypoint2d_local = keypoint2d_local
        return keypoint2d_local

    def update_keypoint3d(self, keypoint3d, crop_center3d, scale3d):
        keypoint_x = (keypoint3d[:, 0] - crop_center3d[0]) * scale3d + self.grid_size // 2
        keypoint_y = (keypoint3d[:, 1] - crop_center3d[1]) * scale3d + self.grid_size // 2
        keypoint_z = (keypoint3d[:, 2] - crop_center3d[2]) * scale3d + self.grid_size // 2
        keypoint3d_local = tf.stack([keypoint_x, keypoint_y, keypoint_z], 1)
        return keypoint3d_local

    @staticmethod
    def project_tf(joint3d, calibK, calibR=None, calibt=None, calibDistCoef=None):
        """ This function projects the 3D hand to 2D using camera parameters
        """
        with tf.name_scope('project_tf'):
            x = joint3d
            if calibR is not None:
                x = tf.matmul(joint3d, calibR, transpose_b=True)
            if calibt is not None:
                x = x + calibt
            xi = tf.divide(x[:, 0], x[:, 2])
            yi = tf.divide(x[:, 1], x[:, 2])

            if calibDistCoef is not None:
                X2 = xi * xi
                Y2 = yi * yi
                XY = X2 * Y2
                R2 = X2 + Y2
                R4 = R2 * R2
                R6 = R4 * R2

                dc = calibDistCoef
                radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
                tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
                tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

                xi = radial * xi + tan_x
                yi = radial * yi + tan_y

            xp = tf.transpose(tf.stack([xi, yi], axis=0))
            pt = tf.matmul(xp, calibK[:2, :2], transpose_b=True) + calibK[:2, 2]
        return pt, x

    @staticmethod
    def switch_joint_order(keypoint, order):
        # reorder the joints to the order used in our network
        assert len(order.shape) == 1, 'order must be 1-dim'
        # axis 0: sample, axis 1: keypoint order, axis 2: xyz
        return keypoint[:, order, ...]

    @staticmethod
    def create_multiple_gaussian_map(coords_wh, output_size, sigma, valid_vec=None, extra=False):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        with tf.name_scope('create_multiple_gaussian_map'):
            sigma = tf.cast(sigma, tf.float32)
            assert len(output_size) == 2
            s = coords_wh.get_shape().as_list()
            coords_wh = tf.cast(coords_wh, tf.int32)
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(coords_wh[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            cond_1_in = tf.logical_and(tf.less(coords_wh[:, 0], output_size[0] - 1), tf.greater(coords_wh[:, 0], 0))
            cond_2_in = tf.logical_and(tf.less(coords_wh[:, 1], output_size[1] - 1), tf.greater(coords_wh[:, 1], 0))
            cond_in = tf.logical_and(cond_1_in, cond_2_in)
            cond = tf.logical_and(cond_val, cond_in)

            coords_wh = tf.cast(coords_wh, tf.float32)

            # create meshgrid
            x_range = tf.expand_dims(tf.range(output_size[0]), 1)
            y_range = tf.expand_dims(tf.range(output_size[1]), 0)

            X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
            Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

            X.set_shape((output_size[0], output_size[1]))
            Y.set_shape((output_size[0], output_size[1]))

            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)

            X_b = tf.tile(X, [1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, s[0]])

            X_b -= coords_wh[:, 0]
            Y_b -= coords_wh[:, 1]

            dist = tf.square(X_b) + tf.square(Y_b)

            scoremap = tf.exp(-dist / (2 * tf.square(sigma))) * tf.cast(cond, tf.float32)

            if extra:
                negative = 1 - tf.reduce_sum(scoremap, axis=2, keep_dims=True)
                negative = tf.minimum(tf.maximum(negative, 0.0), 1.0)
                scoremap = tf.concat([scoremap, negative], axis=2)

            return scoremap

    @staticmethod
    def create_multiple_gaussian_map_3d(keypoint_3d, output_size, sigma3d, valid_vec=None, extra=False):
        """ Creates a 3D heatmap for the hand skeleton
        """
        with tf.name_scope('create_multiple_gaussian_map_3d'):
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(keypoint_3d[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            sigma3d = tf.cast(sigma3d, tf.float32)
            # reverse the order of axis: tensorflow uses NDHWC
            reverse = keypoint_3d[:, ::-1]

            z_range = tf.expand_dims(tf.expand_dims(tf.range(output_size, dtype=tf.float32), 1), 2)
            y_range = tf.expand_dims(tf.expand_dims(tf.range(output_size, dtype=tf.float32), 0), 2)
            x_range = tf.expand_dims(tf.expand_dims(tf.range(output_size, dtype=tf.float32), 0), 1)

            Z = tf.tile(z_range, [1, output_size, output_size])
            Y = tf.tile(y_range, [output_size, 1, output_size])
            X = tf.tile(x_range, [output_size, output_size, 1])

            Z = tf.expand_dims(Z, -1)
            Y = tf.expand_dims(Y, -1)
            X = tf.expand_dims(X, -1)

            s = reverse.get_shape().as_list()

            Z_b = tf.tile(Z, [1, 1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, 1, s[0]])
            X_b = tf.tile(X, [1, 1, 1, s[0]])

            Z_b -= reverse[:, 0]
            Y_b -= reverse[:, 1]
            X_b -= reverse[:, 2]

            dist = tf.square(X_b) + tf.square(Y_b) + tf.square(Z_b)

            scoremap_3d = tf.exp(-dist / (2 * tf.square(sigma3d))) * tf.cast(cond_val, tf.float32)

            if extra:
                negative = 1 - tf.reduce_sum(scoremap_3d, axis=3, keep_dims=True)
                negative = tf.minimum(tf.maximum(negative, 0.0), 1.0)
                scoremap_3d = tf.concat([scoremap_3d, negative], axis=3)

            return scoremap_3d

    def start_from(self, idx):
        for key, value in self.tensor_dict.items():
            if value.size > 0:
                self.tensor_dict[key] = value[idx:]
