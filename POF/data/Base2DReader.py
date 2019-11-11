import tensorflow as tf
from data.BaseReader import BaseReader
import numpy as np


class Base2DReader(BaseReader):
    # inherit from BaseReader, implement different 2D cropping (cropping from 2D)

    def __init__(self, objtype=0, shuffle=True, batch_size=1, crop_noise=False):
        super(Base2DReader, self).__init__(objtype, shuffle, batch_size, crop_noise)

    def get(self, withPAF=True, read_image=True, imw=1920, imh=1080):
        assert type(withPAF) == bool
        assert self.objtype in (0, 1)

        # produce data from slice_input_producer
        flow_list = tf.train.slice_input_producer(list(self.tensor_dict.values()), shuffle=self.shuffle)
        flow_dict = {key: flow_list[ik] for ik, key in enumerate(self.tensor_dict.keys())}

        # build data dictionary
        data_dict = {}
        data_dict['img_dir'] = flow_dict['img_dirs']

        PAF_given = False

        if self.objtype == 0:
            body2d = flow_dict['body']
            data_dict['body_valid'] = flow_dict['body_valid']
            data_dict['keypoint_uv_origin'] = body2d
            if 'body_3d' in flow_dict:
                data_dict['keypoint_xyz_origin'] = flow_dict['body_3d']
                data_dict['keypoint_xyz_local'] = flow_dict['body_3d']
                PAF_given = True
        elif self.objtype == 1:
            cond_left = tf.reduce_any(tf.cast(flow_dict['left_hand_valid'], dtype=tf.bool))  # 0 for right hand, 1 for left hand
            hand2d = tf.cond(cond_left, lambda: flow_dict['left_hand'], lambda: flow_dict['right_hand'])  # in world coordinate
            hand2d = tf.cast(hand2d, tf.float32)
            data_dict['keypoint_uv_origin'] = hand2d
            data_dict['left_hand_valid'] = flow_dict['left_hand_valid']
            data_dict['right_hand_valid'] = flow_dict['right_hand_valid']
            if 'left_hand_3d' in flow_dict and 'right_hand_3d' in flow_dict:
                hand3d = tf.cond(cond_left, lambda: flow_dict['left_hand_3d'], lambda: flow_dict['right_hand_3d'])
                data_dict['keypoint_xyz_origin'] = hand3d
                data_dict['keypoint_xyz_local'] = hand3d
                PAF_given = True

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
        if 'other_bbox' in flow_dict:
            ob = flow_dict['other_bbox']
            Xindmap = tf.tile(tf.expand_dims(tf.range(imw, dtype=tf.int32), 0), [imh, 1])
            Xindmap = tf.tile(tf.expand_dims(Xindmap, 2), [1, 1, 20])
            Yindmap = tf.tile(tf.expand_dims(tf.range(imh, dtype=tf.int32), 1), [1, imw])
            Yindmap = tf.tile(tf.expand_dims(Yindmap, 2), [1, 1, 20])

            x_out = tf.logical_or(tf.less(Xindmap, ob[:, 0]), tf.greater_equal(Xindmap, ob[:, 2]))
            y_out = tf.logical_or(tf.less(Yindmap, ob[:, 1]), tf.greater_equal(Yindmap, ob[:, 3]))
            out = tf.cast(tf.logical_or(x_out, y_out), tf.float32)
            out = tf.reduce_min(out, axis=2)
            mask = tf.minimum(mask, out)
        data_dict['mask'] = mask

        if self.objtype in (0, 1):
            if self.objtype == 0:
                keypoints = body2d
                valid = flow_dict['body_valid']
            elif self.objtype == 1:
                keypoints = hand2d
                body2d = hand2d
                valid = tf.cond(cond_left, lambda: flow_dict['left_hand_valid'], lambda: flow_dict['right_hand_valid'])
                data_dict['hand_valid'] = valid
                if PAF_given:
                    body3d = hand3d

            crop_center2d, scale2d = self.calc_crop_scale2d(keypoints, valid)
            data_dict['crop_center2d'] = crop_center2d
            data_dict['scale2d'] = scale2d

            if self.rotate_augmentation:
                print('using rotation augmentation')
                rotate_angle = tf.random_uniform([], minval=-np.pi * 40 / 180, maxval=np.pi * 40 / 180)
                R2 = tf.reshape(tf.stack([tf.cos(rotate_angle), -tf.sin(rotate_angle), tf.sin(rotate_angle), tf.cos(rotate_angle)]), [2, 2])
                body2d = tf.matmul((body2d - crop_center2d), R2) + crop_center2d
                data_dict['keypoint_uv_origin'] = body2d
                if PAF_given:
                    R3 = tf.reshape(tf.stack([tf.cos(rotate_angle), -tf.sin(rotate_angle), 0., tf.sin(rotate_angle), tf.cos(rotate_angle), 0., 0., 0., 1.]), [3, 3])
                    body3d = tf.matmul(body3d, R3)
                    data_dict['keypoint_xyz_origin'] = body3d
                    data_dict['keypoint_xyz_local'] = body3d
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
                num_keypoint = body2d_local.get_shape().as_list()[0]
                zeros = tf.zeros([num_keypoint, 1], dtype=tf.float32)
                if PAF_given:
                    data_dict['PAF'] = createPAF(body2d_local, body3d, self.objtype, (self.crop_size, self.crop_size), normalize_3d=True, valid_vec=valid)
                    data_dict['PAF_type'] = tf.ones([], dtype=bool)  # 0 for 2D PAF, 1 for 3D PAF
                else:
                    data_dict['PAF'] = createPAF(body2d_local, tf.concat([body2d, zeros], axis=1), self.objtype, (self.crop_size, self.crop_size), normalize_3d=False, valid_vec=valid)
                    data_dict['PAF_type'] = tf.zeros([], dtype=bool)  # 0 for 2D PAF, 1 for 3D PAF

            if self.objtype == 1:  # this is hand, flip the image if it is right hand
                data_dict['image_crop'] = tf.cond(cond_left, lambda: data_dict['image_crop'], lambda: data_dict['image_crop'][:, ::-1, :])
                data_dict['mask_crop'] = tf.cond(cond_left, lambda: data_dict['mask_crop'], lambda: data_dict['mask_crop'][:, ::-1])
                data_dict['scoremap2d'] = tf.cond(cond_left, lambda: data_dict['scoremap2d'], lambda: data_dict['scoremap2d'][:, ::-1, :])
                data_dict['keypoint_uv_local'] = tf.cond(cond_left, lambda: data_dict['keypoint_uv_local'],
                                                         lambda: tf.constant([self.crop_size, 0], tf.float32) + tf.constant([-1, 1], tf.float32) * data_dict['keypoint_uv_local'])
                if withPAF:
                    data_dict['PAF'] = tf.cond(cond_left, lambda: data_dict['PAF'],
                                               lambda: (data_dict['PAF'][:, ::-1, :]) * tf.constant([-1, 1, 1] * (data_dict['PAF'].get_shape().as_list()[2] // 3), dtype=tf.float32))

        names, tensors = zip(*data_dict.items())

        if self.shuffle:
            tensors = tf.train.shuffle_batch_join([tensors],
                                                  batch_size=self.batch_size,
                                                  capacity=100,
                                                  min_after_dequeue=50,
                                                  enqueue_many=False)
        else:
            tensors = tf.train.batch_join([tensors],
                                          batch_size=self.batch_size,
                                          capacity=100,
                                          enqueue_many=False)

        return dict(zip(names, tensors))
