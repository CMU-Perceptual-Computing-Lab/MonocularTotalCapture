import numpy as np
import numpy.linalg as nl
from utils.general import connMat

a4_to_main = {
    'body': np.array([1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16, 19, 20], dtype=np.int64),  # convert to order of openpose
    '1_body': np.array([1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16, 19, 20], dtype=np.int64),  # convert to order of openpose
    '2_body': np.array([1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16, 19, 20], dtype=np.int64),  # convert to order of openpose
    'left_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    '1_left_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    '2_left_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    'right_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    '1_right_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    '2_right_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),  # convert to order of freiburg
    'openpose_lhand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'openpose_rhand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'openpose_lhand_score': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'openpose_rhand_score': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
}

human36m_to_main = {
    'body': np.array([9, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 17, 17, 17, 17, 10, 17], dtype=np.int64)
}

mpi3d_to_main = {
    'body': np.array([6, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 28, 28, 28, 28, 7], dtype=np.int64)
}

adam_to_main = {
    'body': np.array([12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7], dtype=np.int64),
    'select_body_main': np.arange(1, 14, dtype=np.int64)
}

COCO_to_main = {
    'body': np.array([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 19], dtype=np.int64),
    'body_valid': np.array([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 19], dtype=np.int64),
    'all_body': np.array([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 19], dtype=np.int64),
    'all_body_valid': np.array([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 19], dtype=np.int64)
}

SMPL_to_main = {  # actually COCOPLUS regressor to main
    'body': np.array([14, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 16, 15, 18, 17, 13], dtype=np.int64)
}

STB_to_main = {
    'left_hand': np.array([0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int64)
}

MPII_to_main = {
    'body': np.array([16, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 16, 16, 16, 16, 9], dtype=np.int64),
    'body_valid': np.array([16, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 16, 16, 16, 16, 9], dtype=np.int64)
}

tsimon_to_main = {
    'left_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'right_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'left_hand_valid': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'right_hand_valid': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
}

GAnerated_to_main = {
    'left_hand': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'left_hand_valid': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'left_hand_3d': np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17], dtype=np.int64),
    'right_hand': np.arange(21, dtype=np.int64),
    'right_hand_valid': np.arange(21, dtype=np.int64),
    'right_hand_3d': np.arange(21, dtype=np.int64)
}

std_body_size = 267.807
std_hand_size = (82.2705 + 79.8843) / 2


def compute_size(joint3d, type_str):
    """ use this to compute size for scaling: joints are in main order.
    """
    length = 0.0
    for ic, conn in enumerate(connMat[type_str]):
        if type_str == 'body':
            if ic in (2, 3, 5, 6, 8, 9, 11, 12):
                length += nl.norm(joint3d[conn[0]] - joint3d[conn[1]])
        else:
            assert type_str == 'hand'
            length += nl.norm(joint3d[conn[0]] - joint3d[conn[1]])
    return length


def main_to_a4(joint):
    assert joint.shape[0] == 20
    output = np.zeros((21, joint.shape[1]), dtype=joint.dtype)
    for io, ic in enumerate(a4_to_main['body']):
        output[ic, :] = joint[io, :]
    output[2, :] = (output[6, :] + output[12, :]) / 2
    return output


def main_to_a4_hand(joint):
    assert joint.shape[0] == 21
    output = np.zeros(joint.shape, dtype=joint.dtype)
    output[0] = joint[0]
    for i in (1, 5, 9, 13, 17):
        output[i:i + 4] = joint[i + 3:i - 1:-1]
    return output


def assemble_total_3d(body, lhand, rhand):
    len_b = compute_size(body, 'body')
    if len_b > 0:
        sbody = (std_body_size / len_b) * body
    else:
        sbody = body
    len_l = compute_size(lhand, 'hand')
    if len_l > 0:
        slhand = (std_hand_size / len_l) * lhand
    else:
        slhand = lhand
    len_r = compute_size(rhand, 'hand')
    if len_r > 0:
        srhand = (std_hand_size / len_r) * rhand
    else:
        srhand = rhand

    sbody = main_to_a4(sbody)
    slhand = main_to_a4_hand(slhand)
    srhand = main_to_a4_hand(srhand)

    slhand_invalid = (slhand[:, 0] == 0) * (slhand[:, 1] == 0) * (slhand[:, 2] == 0)
    srhand_invalid = (srhand[:, 0] == 0) * (srhand[:, 1] == 0) * (srhand[:, 2] == 0)

    if not slhand[0].any():
        slhand_invalid[:] = True
    if not srhand[0].any():
        srhand_invalid[:] = True

    lhand_idx_a4 = 5
    rhand_idx_a4 = 11

    shift_lhand = sbody[lhand_idx_a4] - slhand[0]
    shift_rhand = sbody[rhand_idx_a4] - srhand[0]

    slhand += shift_lhand
    srhand += shift_rhand

    slhand[slhand_invalid] = 0
    srhand[srhand_invalid] = 0

    return np.concatenate([sbody, slhand, srhand], axis=0), std_body_size / len_b


def assemble_total_2d(body_2d, lhand_2d, rhand_2d):
    keypoint_list = []
    for i, item in enumerate((body_2d, lhand_2d, rhand_2d)):
        keypoint = item['uv_local']
        keypoint = (keypoint - 184) / item['scale2d'] + item['crop_center2d']
        valid = item['valid']
        keypoint = keypoint * np.stack([valid, valid], axis=1)  # remove those invalid values
        if i == 0:
            keypoint = main_to_a4(keypoint)
        else:
            keypoint = main_to_a4_hand(keypoint)
        keypoint_list.append(keypoint)

    ret = np.concatenate(keypoint_list, axis=0)
    ret[np.isnan(ret)] = 0.0  # nan when the whole joint is zero
    return ret


def main_to_human36m(joint):
    # except 9, 10 in human36m
    out = np.zeros((17, 3), dtype=joint.dtype)
    for im, ih in enumerate(human36m_to_main['body']):
        if ih == 17:  # virtual zero joint
            continue
        out[ih] = np.copy(joint[im, :])
    out[0] = (out[1] + out[4]) / 2  # middle hip
    out[7] = (out[1] + out[4] + out[11] + out[14]) / 4  # abdomen (average of l/r hip, l/r shoulder)
    return out
