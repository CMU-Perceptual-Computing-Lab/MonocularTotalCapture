# Run this script with OpenCV2
import cv2
import numpy as np
import os
import json

source_dir = '/media/posefs3b/Users/gines/mpii_mask'
target_dir = '/media/posefs3b/Users/donglaix/mpii_mask'

if __name__ == '__main__':
    path_to_db = './MPII_collected.json'
    with open(path_to_db) as f:
        db_data = json.load(f)
    total_num = len(db_data['img_paths'])
    for i in range(total_num):
        print ('processing image {} / {}'.format(i, total_num))
        bbox = np.array(db_data['bbox'][i], dtype=np.float32)
        bbox_other = np.array(db_data['bbox_other'][i], dtype=np.float32).reshape(-1, 4)
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        img_path = db_data['img_paths'][i]
        source_mask = os.path.join('/media/posefs3b/Users/gines/mpii_mask', img_path)
        mask = cv2.imread(source_mask)
        mask = (mask[:, :, 0] >= 128).astype(np.uint8)  # the stored data are 0 ~ 255, convert to bool
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        belong = []
        for cnt in contours:
            x1 = np.amin(cnt[:, 0, 0])
            x2 = np.amax(cnt[:, 0, 0])
            y1 = np.amin(cnt[:, 0, 1])
            y2 = np.amax(cnt[:, 0, 1])
            if x < x1 or x > x2 or y < y1 or y > y2:
                belong.append(False)
                continue
            # the center is inside this contour, now check the all other bounding boxes
            xo = (bbox_other[:, 0] + bbox_other[:, 2]) / 2
            yo = (bbox_other[:, 1] + bbox_other[:, 3]) / 2
            if ((xo >= x1) * (xo <= x2) * (yo >= y1) * (yo <= y2)).any():  # the center of any other bounding boxes fall inside
                belong.append(False)
            else:
                belong.append(True)  # the center of current bbox is in and others are not in.
        assert len(belong) == len(contours)
        new_mask = np.ones(mask.shape, dtype=np.uint8)
        cv2.drawContours(new_mask, [cnt for TF, cnt in zip(belong, contours) if not TF], -1, 0, -1)
        cv2.imwrite(os.path.join(target_dir, '{:05d}.png'.format(i)), new_mask)
