import pickle
import os
import numpy as np
from utils.general import plot2d_cv2
import cv2

map_index = np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=int)


def project(joints, K, R=None, t=None, distCoef=None):
    """ Perform Projection.
        joints: N * 3
    """
    x = joints.T
    if R is not None:
        x = np.dot(R, x)
    if t is not None:
        x = x + t.reshape(3, 1)

    xp = x[:2, :] / x[2, :]

    if distCoef is not None:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = distCoef
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    pt = np.dot(K[:2, :2], xp) + K[:2, 2].reshape((2, 1))

    return pt.T, x.T


if __name__ == '__main__':
    image_root = '/media/posefs0c/panopticdb/'
    save_root = '/media/posefs1b/Users/donglaix/clean_a4_hand/crop_hand_new/'
    with open('./data/a4_collected.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('./data/camera_data_a4.pkl', 'rb') as f:
        cam_data = pickle.load(f)

    for set_name, set_data in data.items():
        for i, sample_data in enumerate(set_data):
            print ('processing {} {} / {}'.format(set_name, i, len(set_data)))
            seqName = sample_data['seqName']
            frame_str = sample_data['frame_str']
            if 'left_hand' in sample_data:
                joints = np.array(sample_data['left_hand']['landmarks']).reshape(-1, 3)
                joints = joints[map_index]
                count_img = 0
                for c in np.random.permutation(31):
                    if count_img == 3:  # enough
                        break
                    if c not in sample_data['left_hand']['2D']:
                        continue
                    if sum(sample_data['left_hand']['2D'][c]['insideImg']) < 15 or \
                       sum(sample_data['left_hand']['2D'][c]['occluded']) > 5 or (sample_data['left_hand']['2D'][c]['occluded'] == 1):
                        continue
                    count_img += 1
                    joint2d, _ = project(joints, cam_data[seqName][c]['K'], cam_data[seqName][c]['R'], cam_data[seqName][c]['t'], cam_data[seqName][c]['distCoef'])
                    img_name = '{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(image_root, 'a4', seqName, frame_str, c, frame_str)
                    img = cv2.imread(img_name)
                    assert img is not None

                    x1 = np.amin(joint2d[:, 0])
                    x2 = np.amax(joint2d[:, 0])
                    y1 = np.amin(joint2d[:, 1])
                    y2 = np.amax(joint2d[:, 1])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    size = max(x2 - x1, y2 - y1)
                    scale = 200 / (1.5 * size)
                    M = np.array([[scale, 0, (100 - scale * cx)],
                                  [0, scale, (100 - scale * cy)]], dtype=float)
                    target_img = cv2.warpAffine(img, M, (200, 200))
                    tjoint2d = (joint2d - np.array([cx, cy])) * scale + 100
                    plot2d_cv2(target_img, tjoint2d, 'hand', s=3, use_color=True)
                    filename = '{}#{}#left#{:02d}.png'.format(seqName, frame_str, c)
                    cv2.imwrite(os.path.join(save_root, filename), target_img)

            if 'right_hand' in sample_data:
                joints = np.array(sample_data['right_hand']['landmarks']).reshape(-1, 3)
                joints = joints[map_index]
                count_img = 0
                for c in np.random.permutation(31):
                    if count_img == 3:  # enough
                        break
                    if c not in sample_data['right_hand']['2D']:
                        continue
                    if sum(sample_data['right_hand']['2D'][c]['insideImg']) < 15 or \
                       sum(sample_data['right_hand']['2D'][c]['occluded']) > 5 or (sample_data['right_hand']['2D'][c]['occluded'] == 1):
                        continue
                    count_img += 1
                    joint2d, _ = project(joints, cam_data[seqName][c]['K'], cam_data[seqName][c]['R'], cam_data[seqName][c]['t'], cam_data[seqName][c]['distCoef'])
                    img_name = '{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(image_root, 'a4', seqName, frame_str, c, frame_str)
                    img = cv2.imread(img_name)
                    assert img is not None

                    x1 = np.amin(joint2d[:, 0])
                    x2 = np.amax(joint2d[:, 0])
                    y1 = np.amin(joint2d[:, 1])
                    y2 = np.amax(joint2d[:, 1])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    size = max(x2 - x1, y2 - y1)
                    scale = 200 / (1.5 * size)
                    M = np.array([[scale, 0, (100 - scale * cx)],
                                  [0, scale, (100 - scale * cy)]], dtype=float)
                    target_img = cv2.warpAffine(img, M, (200, 200))
                    tjoint2d = (joint2d - np.array([cx, cy])) * scale + 100
                    plot2d_cv2(target_img, tjoint2d, 'hand', s=3, use_color=True)
                    filename = '{}#{}#righ#{:02d}.png'.format(seqName, frame_str, c)
                    cv2.imwrite(os.path.join(save_root, filename), target_img)
