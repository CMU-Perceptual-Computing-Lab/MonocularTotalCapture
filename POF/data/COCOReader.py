import tensorflow as tf
import os
import numpy as np
import json
from data.Base2DReader import Base2DReader
from utils.keypoint_conversion import COCO_to_main, MPII_to_main


class COCOReader(Base2DReader):
    def __init__(self, name='COCO', mode='training', objtype=0, shuffle=True, batch_size=1, crop_noise=False):
        super(COCOReader, self).__init__(objtype, shuffle, batch_size, crop_noise)

        self.name = name
        assert name in ('COCO', 'MPII')
        assert mode in ('training', 'evaluation')

        if name == 'COCO':
            self.image_root = '/media/posefs3b/Users/gines/openpose_train/dataset/COCO/cocoapi/images/train2017/'
            self.mask_root = '/media/posefs3b/Users/gines/openpose_train/dataset/COCO/cocoapi/images/mask2017/train2017/'

            assert mode == 'training'
            path_to_db = '/media/posefs3b/Users/gines/openpose_train/dataset/COCO/json/COCO.json'

            with open(path_to_db) as f:
                db_data = json.load(f)

            img_dirs = []
            mask_dirs = []
            human = {'body': [], 'body_valid': [], 'other_bbox': []}
            for i, image_data in enumerate(db_data['root']):
                # bounding box test
                # discard the image if this bounding box overlaps with any other bounding box
                bbox = np.array(image_data['bbox'], dtype=np.float32)
                bbox[2:] += bbox[:2]

                if type(image_data['bbox_other']) != dict and len(image_data['bbox_other']) > 0:
                    bbox_other = np.array(image_data['bbox_other'], dtype=np.float32).reshape(-1, 4)
                    bbox_other[:, 2:] += bbox_other[:, :2]
                    # xmin = np.maximum(bbox_other[:, 0], bbox[0])
                    # ymin = np.maximum(bbox_other[:, 1], bbox[1])
                    # xmax = np.minimum(bbox_other[:, 2], bbox[2])
                    # ymax = np.minimum(bbox_other[:, 3], bbox[3])
                    # overlap_cond = np.logical_and(xmin < xmax, ymin < ymax).any()
                    # if overlap_cond:
                    #     continue

                    zero_left = np.zeros([20 - bbox_other.shape[0], 4])
                    bbox_other = np.concatenate([bbox_other, zero_left], axis=0).astype(np.int32)

                else:
                    bbox_other = np.zeros([20, 4], dtype=np.int32)

                body = np.array(image_data['joint_self'], dtype=int)
                if np.sum(body[:, 2] == 1) <= 3:
                    continue

                img_dirs.append(os.path.join(self.image_root, image_data['img_paths']))
                mask_dirs.append(os.path.join(self.mask_root, image_data['img_paths'][:-3] + 'png'))

                neck = (body[5:6, :2] + body[6:7, :2]) / 2
                heattop = np.zeros((1, 2), dtype=int)
                chest = 0.25 * (body[5:6, :2] + body[6:7, :2] + body[11:12, :2] + body[12:13, :2])
                neck_valid = np.logical_and(body[5:6, 2] == 1, body[6:7, 2] == 1)
                heattop_valid = np.zeros((1,), dtype=bool)
                chest_valid = np.logical_and(body[5:6, 2] == 1, body[6:7, 2] == 1) * np.logical_and(body[11:12, 2] == 1, body[12:13, 2] == 1)
                body2d = np.concatenate([body[:, :2], neck, heattop, chest], axis=0)
                valid = np.concatenate([body[:, 2] == 1, neck_valid, heattop_valid, chest_valid])

                human['body'].append(body2d.astype(np.float32))
                human['body_valid'].append(valid.astype(bool))
                human['other_bbox'].append(bbox_other)

            human['img_dirs'] = img_dirs
            human['mask_dirs'] = mask_dirs
            order_dict = COCO_to_main

        elif name == 'MPII':
            self.image_root = '/media/posefs3b/Datasets/MPI/images/'
            self.mask_root = '/media/posefs3b/Users/donglaix/mpii_mask/'
            path_to_db = 'data/MPII_collected.json'
            with open(path_to_db) as f:
                db_data = json.load(f)
            total_num = len(db_data['img_paths'])
            human = {'body': [], 'body_valid': [], 'other_bbox': []}
            img_dirs = []
            mask_dirs = []
            for i in range(total_num):
                if (mode == 'training' and not db_data['is_train'][i]) and (mode == 'evaluation' and db_data['is_train'][i]):
                    continue
                body = np.array(db_data['joint_self'][i], dtype=int)
                if np.sum(body[:, 2] == 1) <= 3:
                    continue
                img_dirs.append(os.path.join(self.image_root, db_data['img_paths'][i]))
                mask_dirs.append(os.path.join(self.mask_root, '{:05d}.png'.format(i)))
                body = np.concatenate([body, np.zeros([1, 3], dtype=int)], axis=0)
                human['body'].append(body[:, :2].astype(np.float32))
                human['body_valid'].append(body[:, 2].astype(bool))
            human['img_dirs'] = img_dirs
            human['mask_dirs'] = mask_dirs
            order_dict = MPII_to_main

        else:
            raise NotImplementedError

        self.register_tensor(human, order_dict)
        self.num_samples = len(self.tensor_dict['img_dirs'])

    def get(self):
        if self.name == 'COCO':
            d = super(COCOReader, self).get(withPAF=True, read_image=True, imw=640, imh=640)
        elif self.name == 'MPII':
            d = super(COCOReader, self).get(withPAF=True, read_image=True, imw=1920, imh=1080)
        else:
            raise NotImplementedError
        return d


if __name__ == '__main__':
    dataset = COCOReader(name='COCO', mode='training', shuffle=False, objtype=0, crop_noise=False)
    data_dict = dataset.get()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    import matplotlib.pyplot as plt
    import utils.general
    from utils.PAF import plot_all_PAF, plot_PAF

    for i in range(dataset.num_samples):
        image_crop, image, body2d, body_valid, img_dir, mask, mask_crop, PAF, PAF_type = \
            sess.run([data_dict['image_crop'], data_dict['image'], data_dict['keypoint_uv_local'], data_dict['body_valid'], data_dict['img_dir'], data_dict['mask'],
                      data_dict['mask_crop'], data_dict['PAF'], data_dict['PAF_type']])
        print ('{}: {}'.format(i, img_dir[0].decode()))
        body2d = np.squeeze(body2d)
        body_valid = np.squeeze(body_valid)
        image_crop = np.squeeze((image_crop + 0.5) * 255).astype(np.uint8)
        image = np.squeeze((image + 0.5) * 255).astype(np.uint8)
        mask = np.squeeze(mask)
        mask_crop = np.squeeze(mask_crop)
        PAF = np.squeeze(PAF)

        mask_image = np.stack([mask] * 3, axis=2)
        mask_crop_image = np.stack([mask_crop] * 3, axis=2)

        fig = plt.figure(1)
        ax1 = fig.add_subplot(231)
        plt.imshow(image_crop)
        utils.general.plot2d(ax1, body2d, valid_idx=body_valid)

        ax2 = fig.add_subplot(232)
        plt.imshow(image)

        ax3 = fig.add_subplot(233)
        plt.gray()
        plt.imshow(mask_image)

        ax4 = fig.add_subplot(234)
        plt.gray()
        plt.imshow(mask_crop_image)

        ax5 = fig.add_subplot(235)
        PAF_img, img_z = plot_all_PAF(PAF, 3)
        ax5.imshow(PAF_img)

        ax6 = fig.add_subplot(236)
        ax6.imshow(img_z)

        plt.show()
