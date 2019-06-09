import tensorflow as tf
import numpy as np
import numpy.linalg as nl
import cv2

# in A4 order (SMC)
tbody_connMat = np.array([0, 1, 0, 3, 3, 4, 4, 5, 0, 9, 9, 10, 10, 11, 0, 2, 2, 6, 6, 7, 7, 8, 2, 12, 12, 13, 13, 14, 1, 15, 15, 16, 1, 17, 17, 18, 0, 19, 0, 20, 20, 12, 20, 6])
thand_connMat = np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 0, 9, 9, 10, 10, 11, 11, 12, 0, 13, 13, 14, 14, 15, 15, 16, 0, 17, 17, 18, 18, 19, 19, 20])
total_connMat = np.concatenate([tbody_connMat, thand_connMat + 21, thand_connMat + 42], axis=0).reshape(-1, 2)

connMat = {
    'body': np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17], [1, 18], [1, 19]], dtype=int),
    'hand': np.array([[0, 4], [4, 3], [3, 2], [2, 1], [0, 8], [8, 7], [7, 6], [6, 5], [0, 12], [12, 11], [11, 10], [10, 9],
                      [0, 16], [16, 15], [15, 14], [14, 13], [0, 20], [20, 19], [19, 18], [18, 17]]),
    'total': total_connMat,
    'face': np.array([]),
    'human3.6m': np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])
}
type_strs = ['body', 'hand']


class LearningRateScheduler:
    """
        Provides scalar tensors at certain iteration as is needed for a multistep learning rate schedule.
    """

    def __init__(self, steps, values):
        self.steps = steps
        self.values = values
        assert len(steps) + 1 == len(values), "There must be one more element in value as step."

    def get_lr(self, global_step):
        with tf.name_scope('lr_scheduler'):

            if len(self.values) == 1:  # 1 value -> no step
                learning_rate = tf.constant(self.values[0])
            elif len(self.values) == 2:  # 2 values -> one step
                cond = tf.greater(global_step, self.steps[0])
                learning_rate = tf.where(cond, self.values[1], self.values[0])
            else:  # n values -> n-1 steps
                cond_first = tf.less(global_step, self.steps[0])

                cond_between = list()
                for ind, step in enumerate(range(0, len(self.steps) - 1)):
                    cond_between.append(tf.logical_and(tf.less(global_step, self.steps[ind + 1]),
                                                       tf.greater_equal(global_step, self.steps[ind])))

                cond_last = tf.greater_equal(global_step, self.steps[-1])
                cond_full = [cond_first]
                cond_full.extend(cond_between)
                cond_full.append(cond_last)
                cond_vec = tf.stack(cond_full)
                lr_vec = tf.stack(self.values)
                learning_rate = tf.where(cond_vec, lr_vec, tf.zeros_like(lr_vec))
                learning_rate = tf.reduce_sum(learning_rate)

            return learning_rate


def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image. When factor is not given does an central crop.

    Inputs:
        image: 4D tensor, [batch, height, width, channels] which will be cropped in height and width dimension
        crop_location: tensor, [batch, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 4D tensor, [batch, crop_size, crop_size, channels]
    """
    with tf.name_scope('crop_image_from_xy'):
        s = image.get_shape().as_list()
        assert len(s) == 4, "Image needs to be of shape [batch, width, height, channel]"
        scale = tf.reshape(scale, [-1])
        crop_location = tf.cast(crop_location, tf.float32)
        crop_location = tf.reshape(crop_location, [s[0], 2])
        crop_size = tf.cast(crop_size, tf.float32)

        crop_size_scaled = crop_size / scale
        y1 = crop_location[:, 0] - crop_size_scaled // 2
        y2 = y1 + crop_size_scaled
        x1 = crop_location[:, 1] - crop_size_scaled // 2
        x2 = x1 + crop_size_scaled
        y1 /= s[1]
        y2 /= s[1]
        x1 /= s[2]
        x2 /= s[2]
        boxes = tf.stack([y1, x1, y2, x2], -1)

        crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
        box_ind = tf.range(s[0])
        image_c = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes, box_ind, crop_size, name='crop')
        return image_c


def detect_keypoints2d(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_uv = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_uv[i, 0] = u
        keypoint_uv[i, 1] = v
    return keypoint_uv


def detect_keypoints3d(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    if len(scoremaps.shape) == 5:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 4, "This function was only designed for 3D Scoremaps."
    assert (s[3] < s[2]) and (s[3] < s[1]) and (s[3] < s[0]), "Probably the input is not correct, because [D, H, W, C] is expected."

    keypoint_coords = np.zeros((s[3], 3))
    for i in range(s[3]):
        z, y, x = np.unravel_index(np.argmax(scoremaps[:, :, :, i]), (s[0], s[1], s[2]))
        keypoint_coords[i, 0] = x
        keypoint_coords[i, 1] = y
        keypoint_coords[i, 2] = z
    return keypoint_coords


def plot2d(ax, keypoint, type_str='body', valid_idx=None, color='red', s=10):
    assert len(keypoint.shape) == 2 and keypoint.shape[1] == 2
    if valid_idx is not None:
        plot_point = keypoint[valid_idx, :]
    else:
        plot_point = keypoint
    ax.scatter(plot_point[:, 0], plot_point[:, 1], c=color, s=s)

    for conn in connMat[type_str]:
        coord1 = keypoint[conn[0]]
        coord2 = keypoint[conn[1]]
        if valid_idx is not None and (not valid_idx[conn[0]] or not valid_idx[conn[1]]):
            continue
        coords = np.vstack([coord1, coord2])
        ax.plot(coords[:, 0], coords[:, 1], c=color)


def plot2d_cv2(img, keypoint, type_str='body', valid_idx=None, s=10, use_color=False):
    assert len(keypoint.shape) == 2 and keypoint.shape[1] == 2
    if valid_idx is not None:
        plot_point = keypoint[valid_idx, :]
    else:
        plot_point = keypoint
    for i, kp in enumerate(plot_point):
        x = int(kp[0])
        y = int(kp[1])
        if x == 0 and y == 0:
            continue
        if not use_color:
            cv2.circle(img, (x, y), s, (255, 0, 0), -1)
        else:
            if i <= 4:
                color = (255, 0, 0)
            elif i <= 8:
                color = (0, 255, 0)
            elif i <= 12:
                color = (0, 0, 255)
            elif i <= 16:
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)
            cv2.circle(img, (x, y), s, color, -1)

    for i, conn in enumerate(connMat[type_str]):
        coord1 = keypoint[conn[0]]
        coord2 = keypoint[conn[1]]
        if valid_idx is not None and (not valid_idx[conn[0]] or not valid_idx[conn[1]]):
            continue
        pt1 = (int(coord1[0]), int(coord1[1]))
        pt2 = (int(coord2[0]), int(coord2[1]))
        if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
            continue
        if not use_color:
            cv2.line(img, pt1, pt2, (255, 0, 0), int(s / 2))
        else:
            if i < 4:
                color = (255, 0, 0)
            elif i < 8:
                color = (0, 255, 0)
            elif i < 12:
                color = (0, 0, 255)
            elif i < 16:
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)
            cv2.line(img, pt1, pt2, color, int(s / 2))


def plot3d(ax, keypoint, type_str='body', valid_idx=None, color='red'):
    assert len(keypoint.shape) == 2 and keypoint.shape[1] == 3
    if valid_idx is not None:
        plot_point = keypoint[valid_idx, :]
    else:
        plot_point = keypoint
    ax.scatter(plot_point[:, 0], plot_point[:, 1], plot_point[:, 2], c=color)

    for conn in connMat[type_str]:
        coord1 = keypoint[conn[0]]
        coord2 = keypoint[conn[1]]
        if valid_idx is not None and (not valid_idx[conn[0]] or not valid_idx[conn[1]]):
            continue
        coords = np.vstack([coord1, coord2])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], c=color)


def h36LimbLength(keypoint):
    assert keypoint.shape == (17, 3)
    connections = np.array([[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]])
    Ls = []
    for conn in connections:
        L = nl.norm(keypoint[conn[0]] - keypoint[conn[1]])
        Ls.append(L)
    return np.array(Ls, dtype=np.float32)
