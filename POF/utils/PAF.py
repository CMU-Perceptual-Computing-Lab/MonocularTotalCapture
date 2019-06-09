import tensorflow as tf
import numpy as np
import numpy.linalg as nl
import utils.general
import skimage.feature
import json
import os

PAF_type = 0
allPAFConnection = [[np.array([[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17], [1, 18], [1, 19], [19, 8], [19, 11]]),
                     np.array([[0, 4], [4, 3], [3, 2], [2, 1], [0, 8], [8, 7], [7, 6], [6, 5], [0, 12], [12, 11], [11, 10], [10, 9], [0, 16], [16, 15], [15, 14], [14, 13], [0, 20], [20, 19], [19, 18], [18, 17]])
                     ],  # PAF type 0 (Original Openpose)
                    [np.array([[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17],
                               [1, 0], [0, 14], [0, 15], [14, 16], [15, 17], [1, 18], [2, 4], [5, 7], [8, 4], [11, 7], [8, 10], [11, 13]]),  # augmented PAF
                     np.array([[0, 4], [4, 3], [3, 2], [2, 1], [0, 8], [8, 7], [7, 6], [6, 5], [0, 12], [12, 11], [11, 10], [10, 9], [0, 16], [16, 15], [15, 14], [14, 13], [0, 20], [20, 19], [19, 18], [18, 17]])
                     ]]  # PAF type 1 (My augmented PAF)
PAFConnection = allPAFConnection[PAF_type]

dist_thresh = 8
if os.path.exists('utils/default_PAF_lengths.json'):
    with open('utils/default_PAF_lengths.json', 'r') as f:
        default_PAF_length = json.load(f)


def getValidPAF(valid, objtype, PAFdim):
    # input "valid": a tensor containing bool valid/invalid for each channel
    # input "objtype": 0 for body, 1 for hand (to select PAFConnection)
    with tf.variable_scope('getValidPAF'):
        assert objtype in (0, 1)
        connection = tf.constant(np.repeat(PAFConnection[objtype], PAFdim, axis=0), dtype=tf.int64)
        batch_size = valid.get_shape().as_list()[0]
        PAF_valid = []
        for ib in range(batch_size):
            b_valid = valid[ib, :]
            assert len(b_valid.get_shape().as_list()) == 1
            indexed_valid = tf.gather(b_valid, connection, axis=0)
            PAF_valid.append(tf.logical_and(indexed_valid[:, 0], indexed_valid[:, 1]))
        PAF_valid = tf.stack(PAF_valid, axis=0)
    return PAF_valid


def getValidPAFNumpy(valid, objtype):
    # used in testing time
    # input "valid": a numpy array containing bool valid/invalid for each channel
    # input "objtype": 0 for body, 1 for hand (to select PAFConnection)
    assert objtype in (0, 1)
    connection = PAFConnection[objtype]
    PAF_valid = []
    for conn in connection:
        connection_valid = valid[conn[0]] and valid[conn[1]]
        PAF_valid.append(connection_valid)
    PAF_valid = np.array(PAF_valid, dtype=bool)
    return PAF_valid


def createPAF(keypoint2d, keypoint3d, objtype, output_size, normalize_3d=True, valid_vec=None):
    # objtype: 0: body, 1: hand
    # output_size: (h, w)
    # keypoint2d: (x, y)
    # normalize_3d: if True: set x^2 + y^2 + z^2 = 1; else set x^2 + y^2 = 1
    with tf.variable_scope('createPAF'):
        assert keypoint2d.get_shape().as_list()[0] == keypoint3d.get_shape().as_list()[0]
        assert keypoint2d.get_shape().as_list()[1] == 2
        assert keypoint3d.get_shape().as_list()[1] == 3

        if valid_vec is None:
            valid_vec = tf.ones([keypoint2d.get_shape()[0]], dtype=tf.bool)

        h_range = tf.expand_dims(tf.range(output_size[0]), 1)
        w_range = tf.expand_dims(tf.range(output_size[1]), 0)

        H = tf.cast(tf.tile(h_range, [1, output_size[1]]), tf.float32)
        W = tf.cast(tf.tile(w_range, [output_size[0], 1]), tf.float32)

        PAFs = []
        for ic, conn in enumerate(PAFConnection[objtype]):
            AB = keypoint2d[conn[1]] - keypoint2d[conn[0]]  # joint 0 - > joint 1
            l_AB = tf.sqrt(tf.reduce_sum(tf.square(AB)))
            AB = AB / l_AB

            dx = W - keypoint2d[conn[0], 0]
            dy = H - keypoint2d[conn[0], 1]

            dist = tf.abs(dy * AB[0] - dx * AB[1])  # cross product

            Xmin = tf.minimum(keypoint2d[conn[0], 0], keypoint2d[conn[1], 0]) - dist_thresh
            Xmax = tf.maximum(keypoint2d[conn[0], 0], keypoint2d[conn[1], 0]) + dist_thresh
            Ymin = tf.minimum(keypoint2d[conn[0], 1], keypoint2d[conn[1], 1]) - dist_thresh
            Ymax = tf.maximum(keypoint2d[conn[0], 1], keypoint2d[conn[1], 1]) + dist_thresh

            within_range = tf.cast(W >= Xmin, tf.float32) * tf.cast(W <= Xmax, tf.float32) * tf.cast(H >= Ymin, tf.float32) * tf.cast(H <= Ymax, tf.float32)
            within_dist = tf.cast(dist < dist_thresh, tf.float32)

            mask = within_range * within_dist

            AB3d = (keypoint3d[conn[1]] - keypoint3d[conn[0]])
            if normalize_3d:
                scale = tf.sqrt(tf.reduce_sum(tf.square(AB3d)))
            else:
                scale = tf.sqrt(tf.reduce_sum(tf.square(AB3d[:2])))
            AB3d /= scale
            AB3d = tf.where(tf.is_nan(AB3d), tf.zeros([3], dtype=tf.float32), AB3d)

            cond_valid = tf.logical_and(valid_vec[conn[0]], valid_vec[conn[1]])
            connPAF = tf.cond(cond_valid, lambda: tf.tile(tf.expand_dims(mask, 2), [1, 1, 3]) * AB3d, lambda: tf.zeros((output_size[0], output_size[1], 3), dtype=tf.float32))
            # create the PAF only when both joints are valid

            PAFs.append(connPAF)

        concat_PAFs = tf.concat(PAFs, axis=2)

    return concat_PAFs


def getColorAffinity(v):
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    summed = RY + YG + GC + CB + BM + MR

    v = min(max(v, 0.0), 1.0) * summed
    if v < RY:
        c = (255., 255. * (v / (RY)), 0.)
    elif v < RY + YG:
        c = (255. * (1 - ((v - RY) / (YG))), 255., 0.)
    elif v < RY + YG + GC:
        c = (0. * (1 - ((v - RY) / (YG))), 255., 255. * ((v - RY - YG) / (GC)))
    elif v < RY + YG + GC + CB:
        c = (0., 255. * (1 - ((v - RY - YG - GC) / (CB))), 255.)
    elif v < summed - MR:
        c = (255. * ((v - RY - YG - GC - CB) / (BM)), 0., 255.)
    elif v < summed:
        c = (255., 0., 255. * (1 - ((v - RY - YG - GC - CB - BM) / (MR))))
    else:
        c = (255., 0., 0.)
    return np.array(c)


def plot_PAF(PAF_array):
    # return a 3-channel uint8 np array
    assert len(PAF_array.shape) == 3
    assert PAF_array.shape[2] == 2 or PAF_array.shape[2] == 3
    out = np.zeros((PAF_array.shape[0], PAF_array.shape[1], 3), dtype=np.uint8)

    # 2D PAF: use Openpose Visualization
    x = PAF_array[:, :, 0]
    y = PAF_array[:, :, 1]
    rad = np.sqrt(np.square(x) + np.square(y))
    rad = np.minimum(rad, 1.0)
    a = np.arctan2(-y, -x) / np.pi
    fk = (a + 1.) / 2.
    for i in range(PAF_array.shape[0]):
        for j in range(PAF_array.shape[1]):
            color = getColorAffinity(fk[i, j]) * rad[i, j]
            out[i, j, :] = color

    if PAF_array.shape[2] == 3:
        # also return the average z value (for judge pointing out / in)

        # total_rad = np.sqrt(np.sum(np.square(PAF_array), axis=2))
        # rz = PAF_array[:, :, 2] / total_rad
        # rz[np.isnan(rz)] = 0.0
        # rz[total_rad < 0.5] = 0.0

        # z_map = np.zeros((PAF_array.shape[0], PAF_array.shape[1], 3))
        # z_map[:, :, 0] = 255 * rz * (rz > 0)
        # z_map[:, :, 1] = 255 * (-rz) * (rz < 0)

        rz = PAF_array[:, :, 2]

        z_map = np.zeros((PAF_array.shape[0], PAF_array.shape[1], 3))
        z_map[:, :, 0] = 255 * rz * (rz > 0)
        z_map[:, :, 1] = 255 * (-rz) * (rz < 0)
        z_map = np.maximum(np.minimum(z_map, 255), 0)
        return out, z_map.astype(np.uint8)

    return out


def plot_all_PAF(PAF_array, PAFdim):
    assert PAFdim in (2, 3)
    if PAFdim == 2:
        assert PAF_array.shape[2] % 2 == 0
        total_PAF_x = np.sum(PAF_array[:, :, ::2], axis=2)
        total_PAF_y = np.sum(PAF_array[:, :, 1::2], axis=2)
        total_PAF = np.stack([total_PAF_x, total_PAF_y], axis=2)
        return plot_PAF(total_PAF)
    else:
        assert PAFdim == 3 and PAF_array.shape[2] % 3 == 0
        total_PAF_x = np.sum(PAF_array[:, :, ::3], axis=2)
        total_PAF_y = np.sum(PAF_array[:, :, 1::3], axis=2)
        total_PAF_z = np.sum(PAF_array[:, :, 2::3], axis=2)
        total_PAF = np.stack([total_PAF_x, total_PAF_y, total_PAF_z], axis=2)
        return plot_PAF(total_PAF)


def PAF_to_3D(coord2d, PAF, objtype=0):
    if objtype == 0:
        depth_root_idx = 1  # put neck at 0-depth
    else:
        assert objtype == 1
        depth_root_idx = 0
    assert len(coord2d.shape) == 2 and coord2d.shape[1] == 2
    coord3d = np.zeros((coord2d.shape[0], 3), dtype=coord2d.dtype)

    coord3d[:, :2] = coord2d
    coord3d[depth_root_idx, 2] = 0.0
    vec3d_array = []
    for ic, conn in enumerate(PAFConnection[objtype]):
        if objtype == 0:
            if PAF_type == 0:
                if ic in (9, 13):
                    continue
            elif PAF_type == 1:
                if ic in (9, 13) or ic >= 20:
                    continue
        A = coord2d[conn[0]]
        B = coord2d[conn[1]]
        u = np.linspace(0.0, 1.0, num=11)
        v = 1.0 - u
        points = (np.outer(A, v) + np.outer(B, u)).astype(int)  # 2 * N
        vec3ds = PAF[points[1], points[0], 3 * ic:3 * ic + 3]  # note order of y, x in index

        vec3d = np.mean(vec3ds, axis=0)
        vec3d[np.isnan(vec3d)] = 0.0  # numerical stability

        if (A == B).all():  # A and B actually coincides with each other, put the default bone length.
            coord3d[conn[1], 0] = A[0]
            coord3d[conn[1], 1] = A[1]
            if vec3d[2] >= 0:
                coord3d[conn[1], 2] = coord3d[conn[0], 2] + default_PAF_length[objtype][ic]
            else:
                coord3d[conn[1], 2] = coord3d[conn[0], 2] - default_PAF_length[objtype][ic]

        else:
            # find the least square solution of Ax = b
            A = np.zeros([3, 2])
            A[2, 0] = -1.
            A[:, 1] = vec3d
            b = coord3d[conn[1]] - coord3d[conn[0]]  # by this time the z-value of target joint should be 0
            x, _, _, _ = nl.lstsq(A, b, rcond=-1)

            if x[1] < 0:  # the direction is reversed
                if vec3d[2] >= 0:
                    coord3d[conn[1], 2] = coord3d[conn[0], 2] + default_PAF_length[objtype][ic]  # assume that this connection is vertical to the screen
                else:
                    coord3d[conn[1], 2] = coord3d[conn[0], 2] - default_PAF_length[objtype][ic]
            else:
                coord3d[conn[1], 2] = x[0]
            if nl.norm(vec3d) < 0.1 or x[1] < 0:  # If there is almost no response, or the direction is reversed, put it zero so that Adam does not fit.
                vec3d[:] = 0

        vec3d_array.append(vec3d)

    return coord3d, np.array(vec3d_array)


def collect_PAF_vec(coord2d, PAF, objtype=0):
    assert len(coord2d.shape) == 2 and coord2d.shape[1] == 2
    assert len(PAF.shape) == 3  # H, W, C
    vec3d_array = []
    for ic, conn in enumerate(PAFConnection[objtype]):
        if objtype == 0:
            if PAF_type == 0 and ic in (9, 13):
                continue
            elif PAF_type == 1 and ic in (9, 13):  # need the extra PAFs here
                continue
        A = coord2d[conn[0]]
        B = coord2d[conn[1]]
        u = np.linspace(0.0, 1.0, num=11)
        v = 1.0 - u
        points = (np.outer(A, v) + np.outer(B, u)).astype(int)  # 2 * N
        if 3 * ic < PAF.shape[2]:   # to be compatible with old network with only 20 PAFs instead of 23
            vec3ds = PAF[points[1], points[0], 3 * ic:3 * ic + 3]  # note order of y, x in index
            vec3d = np.mean(vec3ds, axis=0)
        else:
            vec3d = np.zeros((3,))
        vec3d[np.isnan(vec3d)] = 0.0  # numerical stability
        vec3d_array.append(vec3d)

    return np.array(vec3d_array)


def recon_skeleton_PAF(vec3ds, objtype=0):
    # reconstruct a skeleton with standard bone length from PAF only
    selected_PAF_array = []
    if objtype == 0:
        coord3d_pred_v = np.zeros([19, 3], dtype=vec3ds.dtype)
        root_idx = 1
    else:
        assert objtype == 1
        coord3d_pred_v = np.zeros([21, 3], dtype=vec3ds.dtype)
        root_idx = 0
    coord3d_pred_v[root_idx] = 0.0
    count_vec = 0
    for ic, conn in enumerate(PAFConnection[objtype]):
        if objtype == 0:
            if PAF_type == 0 and (ic in (9, 13) or ic >= 21):
                continue
            elif PAF_type == 1 and ic in (9, 13):
                continue
        vec = vec3ds[count_vec]
        vlength = nl.norm(vec)
        assert vlength > 0
        if vlength < 0.1:  # almost no response, set to 0
            vec = np.zeros(3, dtype=vec3ds.dtype)
        else:
            vec = vec / vlength  # unit vector
        selected_PAF_array.append(vec)
        count_vec += 1
        if objtype == 0 and PAF_type == 1 and ic >= 20:
            continue
        coord3d_pred_v[conn[1]] = coord3d_pred_v[conn[0]] + default_PAF_length[objtype][ic] * vec

    return coord3d_pred_v, np.array(selected_PAF_array)


def connection_score_2d(A, B, PAF):
    AB = (B - A).astype(np.float32)
    if not AB.any():
        # A B coincides
        return 0.1
    AB /= nl.norm(AB.astype(np.float32))
    s = PAF.shape
    assert len(s) == 3
    u = np.linspace(0.0, 1.0, num=11)
    v = 1.0 - u
    points = (np.outer(A, v) + np.outer(B, u)).astype(int)
    vec2ds = PAF[points[1], points[0], :2]
    inner_product = np.dot(vec2ds, AB)
    return np.mean(inner_product)


def detect_keypoints2d_PAF(scoremaps, PAF, objtype=0, weight_conn=1.0, mean_shift=False, prev_frame=None):
    print('PAF_type {}'.format(PAF_type))
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    num_candidate = 5
    local_maxs = []
    for i in range(s[2]):
        candidates = skimage.feature.peak_local_max(scoremaps[:, :, i], num_peaks=num_candidate)
        if candidates.shape[0] < num_candidate:
            # if less than that, replicate the first element
            if candidates.shape[0] > 0:
                candidates = np.concatenate([candidates[0][np.newaxis, :]] * (num_candidate - candidates.shape[0]) + [candidates], axis=0)
            else:
                candidates = np.zeros((5, 2), dtype=int)
        local_maxs.append(candidates)

    if objtype == 0:
        root_idx = 1  # starting constructing the tree from root_idx
    else:
        assert objtype == 1
        root_idx = 0
    joint_idx_list = [root_idx]
    candidate_idx_list = [[c] for c in range(num_candidate)]
    sum_score_list = [scoremaps[local_maxs[root_idx][c, 0], local_maxs[root_idx][c, 1], root_idx] for c in range(num_candidate)]
    if prev_frame is not None:
        for c in range(num_candidate):
            sum_score_list[c] -= 20 * nl.norm(local_maxs[root_idx][candidate_idx_list[c][0]][::-1] - prev_frame[c]) / (s[0] + s[1])

    # dynamic programming
    for iconn, conn in enumerate(PAFConnection[objtype]):
        if objtype == 0:
            if PAF_type == 0:
                if iconn in (9, 13) or iconn >= 21:  # unused PAF connection
                    continue
            elif PAF_type == 1:
                if iconn in (9, 13) or iconn >= 20:
                    continue
        joint_idx_list.append(conn[1])
        candidates = local_maxs[conn[1]]
        new_candidate_idx_list = []
        new_sum_score_list = []
        for ican, candidate in enumerate(candidates):
            best_sum_score = -np.inf
            best_candidate_idx = None
            B = candidate[::-1]
            for candidate_idx, sum_score in zip(candidate_idx_list, sum_score_list):
                parent_idx = conn[0]
                parent_candidate_idx = candidate_idx[joint_idx_list.index(parent_idx)]
                A = local_maxs[parent_idx][parent_candidate_idx][::-1]
                connection_score = connection_score_2d(A, B, PAF[:, :, 3 * iconn:3 * iconn + 3])
                new_sum_score = sum_score + scoremaps[candidate[0], candidate[1], conn[1]] + weight_conn * connection_score  # TODO
                if prev_frame is not None:
                    new_sum_score -= 20 * nl.norm(prev_frame[conn[1]] - B) / (s[0] + s[1])
                if new_sum_score > best_sum_score:
                    best_sum_score = new_sum_score
                    best_candidate_idx = candidate_idx
            assert best_candidate_idx is not None

            new_sum_score_list.append(best_sum_score)
            new_candidate_idx_list.append(best_candidate_idx + [ican])

        sum_score_list = new_sum_score_list
        candidate_idx_list = new_candidate_idx_list

    best_candidate_idx = candidate_idx_list[np.argmax(sum_score_list)]
    best_candidate_idx_joint_order = np.zeros_like(best_candidate_idx)
    best_candidate_idx_joint_order[np.array(joint_idx_list, dtype=int)] = best_candidate_idx
    best_candidate = np.array([local_maxs[i][j] for i, j in enumerate(best_candidate_idx_joint_order)])
    coord2d = best_candidate[:, ::-1]
    if objtype == 0:
        assert coord2d.shape[0] == 19 or coord2d.shape[0] == 20
    if objtype == 1:
        assert coord2d.shape[0] == 21
    scores = []
    for i in range(coord2d.shape[0]):
        scores.append(scoremaps[coord2d[i, 1], coord2d[i, 0], i])

    if mean_shift:
        dWidth = 3
        dHeight = 3
        new_coord2d = []
        for i in range(coord2d.shape[0]):
            x1 = max(coord2d[i, 0] - dWidth, 0)
            x2 = min(coord2d[i, 0] + dWidth + 1, s[1])
            y1 = max(coord2d[i, 1] - dHeight, 0)
            y2 = min(coord2d[i, 1] + dHeight + 1, s[0])
            Xmap = np.arange(x1, x2)
            Ymap = np.arange(y1, y2)
            local_scoremap = scoremaps[y1:y2, x1:x2, i]
            gt0 = (local_scoremap > 0)
            if gt0.any():
                pos_scoremap = gt0 * local_scoremap
                xAcc = np.sum(pos_scoremap * Xmap)
                yAcc = np.sum(np.transpose(pos_scoremap) * Ymap)
                scoreAcc = np.sum(pos_scoremap)
                new_coord2d.append([xAcc / scoreAcc, yAcc / scoreAcc])
            else:
                new_coord2d.append([coord2d[i, 0], coord2d[i, 1]])
        coord2d = np.array(new_coord2d, dtype=np.float32)
    return coord2d.astype(np.float32), np.array(scores, dtype=np.float32)


"""
Tensorized get_color_affinity()
RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
summed = RY + YG + GC + CB + BM + MR

v = torch.clamp(v, min=0., max=1.) * summed
# v = min(max(v, 0.0), 1.0) * summed
value = v.cpu().detach().numpy() # [O, H, W]
O, H, W = value.shape
record = np.zeros([O, H, W])
out = np.zeros([O, H, W, 3], dtype=value.dtype)
out[:, :, :, 0] = 255.
print(out.shape)
# if v < RY:
# c = (255., 255. * (v / (RY)), 0.)
idx = np.where(np.logical_and(value < RY, record == 0))
record[idx] = 1
idx_ext = idx + (np.array([1] * len(idx[0])),)
out[idx_ext] = 255. * value[idx] / RY

# elif v < RY + YG:
# c = (255. * (1 - ((v - RY) / (YG))), 255., 0.)
idx = np.where(np.logical_and(value < RY + YG, record == 0))
record[idx] = 1
idx_ext = idx + (np.array([0] * len(idx[0])),)
out[idx_ext] = 255. * (1 - ((value[idx] - RY) / (YG)))
idx_ext = idx + (np.array([1] * len(idx[0])),)
out[idx_ext] = 255.

# elif v < RY + YG + GC:
# c = (0. * (1 - ((v - RY) / (YG))), 255., 255. * ((v - RY - YG) / (GC)))
idx = np.where(np.logical_and(value < RY + YG + GC, record == 0))
record[idx] = 1
idx_ext = idx + (np.array([0] * len(idx[0])),)
out[idx_ext] = 0.
idx_ext = idx + (np.array([1] * len(idx[0])),)
out[idx_ext] = 255
idx_ext = idx + (np.array([2] * len(idx[0])),)
out[idx_ext] = 255. * ((value[idx] - RY - YG) / (GC))

# elif v < RY + YG + GC + CB:
# c = (0., 255. * (1 - ((v - RY - YG - GC) / (CB))), 255.)
idx = np.where(np.logical_and(value < RY + YG + GC + CB, record == 0))
record[idx] = 1
idx_ext = idx + (np.array([0] * len(idx[0])),)
out[idx_ext] = 0.
idx_ext = idx + (np.array([1] * len(idx[0])),)
out[idx_ext] = 255. * (1 - ((value[idx] - RY - YG - GC) / (CB)))
idx_ext = idx + (np.array([2] * len(idx[0])),)
out[idx_ext] = 255.

# elif v < summed - MR:
# c = (255. * ((v - RY - YG - GC - CB) / (BM)), 0., 255.)
idx = np.where(np.logical_and(value < summed - MR, record == 0))
record[idx] = 1
idx_ext = idx + (np.array([0] * len(idx[0])),)
out[idx_ext] = 255.
"""
