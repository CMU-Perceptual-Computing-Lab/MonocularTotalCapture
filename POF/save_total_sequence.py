from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D
import argparse
import cv2
import os
from time import time
import json

from nets.CPM import CPM
from utils.load_ckpt import load_weights_from_snapshot
import utils.general
import utils.keypoint_conversion
import utils.PAF
import pickle
from utils.smoothing import savitzky_golay

body_zoom = 1.8
# hand_zoom = 2.5  # dslr_hands5, dslr_hands6, youtube_talkshow1
hand_zoom = 1.5  # youtube_conduct4
TRACK_HAND = True
BACK_TRACK_THRESH = 2.0

# evaluate both hands and body
parser = argparse.ArgumentParser()
parser.add_argument('--visualize', '-v', action='store_true')
parser.add_argument('--seqName', '-s', type=str)
parser.add_argument('--path', '-p', type=str)
parser.add_argument('--start-from', type=int, default=1)
parser.add_argument('--end-index', type=int, default=-1)
parser.add_argument('--width', type=int, default=1920)  # to determine whether a keypoint is out of image
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--save-image', action='store_true')
parser.add_argument('--freeze', '-f', action='store_true')  # upperbody only
args = parser.parse_args()

assert os.path.isdir(args.path)
if not os.path.isdir(os.path.join(args.path, 'net_output')):
    os.makedirs(os.path.join(args.path, 'net_output'))
assert os.path.isdir(os.path.join(args.path, 'net_output'))
if args.save_image:
    for folder in ['/body_2d', '/lhand_2d', '/rhand_2d', '/paf_xy_body', '/paf_z_body', '/paf_xy_lhand', '/paf_z_lhand', '/paf_xy_rhand', '/paf_z_rhand', '/heatmap']:
        try:
            os.makedirs(args.path + folder)
        except Exception as e:
            print ('Folder {} exists'.format(args.path + folder))

start_from = args.start_from
end_index = args.end_index
image_root = os.path.join(args.path, 'raw_image')
pkl_file = os.path.join(args.path, '{}.pkl'.format(args.seqName))
with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)
num_samples = len(pkl_data[0])  # number of frames collected in pkl
K = np.array(pkl_data[5]['K'], dtype=np.float32)
s = [1, 368, 368, 3]
assert s[1] == s[2]
data = {
    'bimage_crop': tf.placeholder_with_default(tf.zeros([s[0], s[1], s[2], 3], dtype=tf.float32),
                                               shape=[s[0], s[1], s[2], 3]),
    'limage_crop': tf.placeholder_with_default(tf.zeros([s[0], s[1], s[2], 3], dtype=tf.float32),
                                               shape=[s[0], s[1], s[2], 3]),
    'rimage_crop': tf.placeholder_with_default(tf.zeros([s[0], s[1], s[2], 3], dtype=tf.float32),
                                               shape=[s[0], s[1], s[2], 3])
}

bcrop_center2d_origin = np.zeros((num_samples, 2), dtype=np.float32)
bscale2d_origin = np.zeros((num_samples,), dtype=np.float32)
# precompute the body bounding box for smoothing
for i in range(num_samples):
    openpose_body = pkl_data[0][i, list(range(18)) + [1, 1], :2].astype(np.float32)  # duplicate neck for headtop and chest
    openpose_body_score = pkl_data[0][i, list(range(18)) + [0, 0], 2].astype(np.float32)
    openpose_body_valid = (openpose_body_score > 0.01)
    if not openpose_body_valid.any():
        # no bounding box
        if i > 0:
            bcrop_center2d_origin[i] = bcrop_center2d_origin[i - 1]
            bscale2d_origin[i] = bscale2d_origin[i - 1]
    min_coord = np.amin(openpose_body[openpose_body_valid], axis=0)
    max_coord = np.amax(openpose_body[openpose_body_valid], axis=0)
    bcrop_center2d_origin[i] = 0.5 * (min_coord + max_coord)
    fit_size = np.amax(np.maximum(max_coord - bcrop_center2d_origin[i], bcrop_center2d_origin[i] - min_coord))
    # if (not openpose_body_valid[9]) and (not openpose_body_valid[10]) and (not openpose_body_valid[12]) and (not openpose_body_valid[13]):
    if args.freeze or ((not openpose_body_valid[9]) and (not openpose_body_valid[10]) and (not openpose_body_valid[12]) and (not openpose_body_valid[13])):
        # upper body only (detected by openpose)
        # crop_size_best = 2 * fit_size * 3  # youtube_talkshow1
        crop_size_best = 2 * fit_size * 4
    else:
        crop_size_best = 2 * fit_size * body_zoom
    bscale2d_origin[i] = float(s[1]) / crop_size_best
bcrop_center2d_smooth = np.stack((savitzky_golay(bcrop_center2d_origin[:, 0], 21, 3), savitzky_golay(bcrop_center2d_origin[:, 1], 21, 3)), axis=1)
bscale2d_smooth = savitzky_golay(bscale2d_origin, 21, 3)
####
print('set bscale2d constant')
# bscale2d_smooth[1:] = bscale2d_smooth[0]
bscale2d_smooth[:-1] = bscale2d_smooth[-1]
if args.visualize:
    plt.plot(bcrop_center2d_origin[:, 0])
    plt.plot(bcrop_center2d_smooth[:, 0])
    plt.show()
    plt.plot(bcrop_center2d_origin[:, 1])
    plt.plot(bcrop_center2d_smooth[:, 1])
    plt.show()
    plt.plot(bscale2d_origin)
    plt.plot(bscale2d_smooth)
    plt.show()

max_rsize = 0.0
max_lsize = 0.0
rhand_ref_frame = -1
lhand_ref_frame = -1
for i in range(num_samples):
    openpose_rhand = pkl_data[2][i, utils.keypoint_conversion.a4_to_main['openpose_rhand'], :2].astype(np.float32)  # duplicate neck for headtop
    openpose_rhand_score = pkl_data[2][i, utils.keypoint_conversion.a4_to_main['openpose_rhand_score'], 2].astype(np.float32)
    openpose_rhand_valid = (openpose_rhand_score > 0.01)
    if openpose_rhand_valid.any():
        min_coord = np.amin(openpose_rhand[openpose_rhand_valid], axis=0)
        max_coord = np.amax(openpose_rhand[openpose_rhand_valid], axis=0)
        rfit_size = np.amax(max_coord - min_coord) / 2
        if rfit_size > max_rsize:
            max_rsize = rfit_size
            rhand_ref_frame = i
    openpose_lhand = pkl_data[1][i, utils.keypoint_conversion.a4_to_main['openpose_lhand'], :2].astype(np.float32)  # duplicate neck for headtop
    openpose_lhand_score = pkl_data[1][i, utils.keypoint_conversion.a4_to_main['openpose_lhand_score'], 2].astype(np.float32)
    openpose_lhand_valid = (openpose_lhand_score > 0.01)
    if openpose_lhand_valid.any():
        min_coord = np.amin(openpose_lhand[openpose_lhand_valid], axis=0)
        max_coord = np.amax(openpose_lhand[openpose_lhand_valid], axis=0)
        lfit_size = np.amax(max_coord - min_coord) / 2
        if lfit_size > max_lsize:
            max_lsize = lfit_size
            lhand_ref_frame = i
assert max_rsize > 0
assert max_lsize > 0
rscale2d_ref = float(s[1]) / (2 * max_rsize * hand_zoom)
lscale2d_ref = float(s[1]) / (2 * max_lsize * hand_zoom)

bodynet = CPM(out_chan=21, crop_size=368, withPAF=True, PAFdim=3, numPAF=23)
handnet = CPM(out_chan=22, numPAF=20, crop_size=368, withPAF=True, PAFdim=3)

with tf.variable_scope('body'):
    # feed through network
    bheatmap_2d, _, bPAF = bodynet.inference(data['bimage_crop'], train=False)
with tf.variable_scope('hand', reuse=tf.AUTO_REUSE):
    lheatmap_2d, _, lPAF = handnet.inference(data['limage_crop'], train=False)
    # rheatmap_2d, _, rPAF = handnet.inference(data['rimage_crop'], train=False)
    rheatmap_2d, _, rPAF = handnet.inference(data['rimage_crop'][:, :, ::-1, :], train=False)  # flip right to left

s = data['bimage_crop'].get_shape().as_list()
data['bheatmap_2d'] = tf.image.resize_images(bheatmap_2d[-1], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)
data['bPAF'] = tf.image.resize_images(bPAF[-1], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)
s = data['limage_crop'].get_shape().as_list()
data['lheatmap_2d'] = tf.image.resize_images(lheatmap_2d[-1], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)
data['lPAF'] = tf.image.resize_images(lPAF[-1], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)
s = data['rimage_crop'].get_shape().as_list()
data['rheatmap_2d'] = tf.image.resize_images(rheatmap_2d[-1][:, :, ::-1, :], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)  # flip back to right hand
data['rPAF'] = tf.image.resize_images(rPAF[-1][:, :, ::-1, :], (s[1], s[2]), tf.image.ResizeMethod.BICUBIC)
data['rPAF'] = data['rPAF'] * tf.constant([-1, 1, 1] * (60 // 3), dtype=tf.float32)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

cpt = './snapshots/Final_qual_domeCOCO_chest_noPAF2D/model-390000'
load_weights_from_snapshot(sess, cpt, discard_list=['Adam', 'global_step', 'beta'], rename_dict={'CPM': 'body/CPM'})
cpt = './snapshots/Final_qual_hand_clear_zoom/model-160000'
load_weights_from_snapshot(sess, cpt, discard_list=['Adam', 'global_step', 'beta'], rename_dict={'CPM': 'hand/CPM'})

eval_list = ['bimage_crop', 'image', 'bcrop_center2d', 'bscale2d', 'bheatmap_2d', 'bPAF', 'body_uv_local', 'img_dir']
eval_list += ['limage_crop', 'lcrop_center2d', 'lscale2d', 'lheatmap_2d', 'lPAF', 'lhand_uv_local']
eval_list += ['rimage_crop', 'rcrop_center2d', 'rscale2d', 'rheatmap_2d', 'rPAF', 'rhand_uv_local']
eval_list += ['K', 'openpose_face', 'body_valid', 'left_hand_valid', 'right_hand_valid', 'openpose_body_score', 'openpose_lhand_score', 'openpose_rhand_score', 'openpose_face_score']
eval_list += ['openpose_foot', 'openpose_foot_score']

BODY_PAF_SELECT_INDEX = np.concatenate([np.arange(9), np.arange(10, 13), np.arange(14, 23)], axis=0)
lcrop_center2d_origin = np.zeros((num_samples, 2), dtype=np.float32)
lscale2d_origin = np.zeros((num_samples), dtype=np.float32)
rcrop_center2d_origin = np.zeros((num_samples, 2), dtype=np.float32)
rscale2d_origin = np.zeros((num_samples), dtype=np.float32)

frame_indices = pkl_data[6]
for i, frame_index in enumerate(frame_indices):
    if frame_index < start_from:
        continue
    if args.end_index > 0 and frame_index > args.end_index:
        break
    if frame_index == start_from:
        start_i = i
    print('Start running frame No. {:08d}'.format(frame_index))
   
    # read the data here
    filename = os.path.join(image_root, pkl_data[4][i])
    image_v = cv2.imread(filename)[:, :, ::-1]  # convert to RGB order
    val_dict = {}
    openpose_body = pkl_data[0][i, list(range(18)) + [1, 1], :2].astype(np.float32)  # duplicate neck for headtop and chest
    openpose_body_score = pkl_data[0][i, list(range(18)) + [0, 0], 2].astype(np.float32)
    openpose_body_valid = (openpose_body_score > 0)
    val_dict['openpose_body'] = openpose_body
    val_dict['openpose_body_score'] = openpose_body_score
    val_dict['openpose_body_valid'] = openpose_body_valid
    val_dict['openpose_face'] = pkl_data[3][i, :, :2]
    val_dict['openpose_face_score'] = pkl_data[3][i, :, 2]
    val_dict['openpose_foot'] = pkl_data[0][i, 18:, :2]
    val_dict['openpose_foot_score'] = pkl_data[0][i, 18:, 2]

    """
    crop body and feed into network
    """
    val_dict['bcrop_center2d'] = bcrop_center2d_smooth[i]
    val_dict['bscale2d'] = bscale2d_smooth[i]
    bcrop_center2d = bcrop_center2d_smooth[i]
    bscale2d = bscale2d_smooth[i]
    # compute the Homography
    bH = np.array([[bscale2d, 0, s[2] / 2 - bscale2d * bcrop_center2d[0]], [0, bscale2d, s[1] / 2 - bscale2d * bcrop_center2d[1]]], dtype=np.float32)
    bimage_crop_v = cv2.warpAffine(image_v, bH, (s[2], s[1]), flags=cv2.INTER_LANCZOS4)
    bimage_crop_v_feed = np.expand_dims((bimage_crop_v / 255 - 0.5), axis=0)

    bheatmap_2d, bPAF = [np.squeeze(_) for _ in sess.run([data['bheatmap_2d'], data['bPAF']], feed_dict={data['bimage_crop']: bimage_crop_v_feed})]
    val_dict['bheatmap_2d'] = bheatmap_2d
    val_dict['bPAF'] = bPAF

    # store the wrist coordinate of previous frame, to help verify hand bounding boxes
    if frame_index > start_from:
        lwrist_last = borigin[7]
        lwrist_valid_last = body_valid[7]
        rwrist_last = borigin[4]
        rwrist_valid_last = body_valid[4]

    # 2D body detection
    if frame_index == start_from or args.seqName == 'qualitative':
        body2d_pred_v, bscore = utils.PAF.detect_keypoints2d_PAF(val_dict['bheatmap_2d'], val_dict['bPAF'])
    else:
        body2d_pred_v, bscore = utils.PAF.detect_keypoints2d_PAF(val_dict['bheatmap_2d'], val_dict['bPAF'], prev_frame=prev_frame)
    prev_frame = body2d_pred_v
    body2d_pred_v = body2d_pred_v[:20, :]   # with chest
    body_valid = (bscore > 0.30)
    body2d_pred_v[np.logical_not(body_valid)] = 0  # must do this, otherwise PAF_to_3D error
    borigin = (body2d_pred_v - 184) / val_dict['bscale2d'] + val_dict['bcrop_center2d']
    bout = (borigin[:, 0] < 0) + (borigin[:, 1] < 0) + (borigin[:, 0] >= args.width) + (borigin[:, 1] >= args.height)
    body2d_pred_v[bout] = 0.0
    body_valid[bout] = False

    # store the wrist coordinate of current frame, to help verify hand bounding boxes
    if frame_index > start_from:
        lwrist = borigin[7]
        lwrist_valid = body_valid[7]
        rwrist = borigin[4]
        rwrist_valid = body_valid[4]

    """
    crop hands and feed into network
    """
    openpose_rhand = pkl_data[2][i, utils.keypoint_conversion.a4_to_main['openpose_rhand'], :2].astype(np.float32)  # duplicate neck for headtop
    openpose_rhand_score = pkl_data[2][i, utils.keypoint_conversion.a4_to_main['openpose_rhand_score'], 2].astype(np.float32)
    openpose_rhand_valid = (openpose_rhand_score > 0.01)
    openpose_lhand = pkl_data[1][i, utils.keypoint_conversion.a4_to_main['openpose_lhand'], :2].astype(np.float32)  # duplicate neck for headtop
    openpose_lhand_score = pkl_data[1][i, utils.keypoint_conversion.a4_to_main['openpose_lhand_score'], 2].astype(np.float32)
    openpose_lhand_valid = (openpose_lhand_score > 0.01)
    val_dict['openpose_rhand'] = openpose_rhand
    val_dict['openpose_rhand_score'] = openpose_rhand_score
    val_dict['openpose_rhand_valid'] = openpose_rhand_valid
    val_dict['openpose_lhand'] = openpose_lhand
    val_dict['openpose_lhand_score'] = openpose_lhand_score
    val_dict['openpose_lhand_valid'] = openpose_lhand_valid

    lscale2d = lscale2d_ref
    rscale2d = rscale2d_ref

    if not TRACK_HAND or frame_index == start_from:   # the first frame
        if openpose_rhand_valid.any():
            min_coord_rhand = np.amin(openpose_rhand[openpose_rhand_valid], axis=0)
            max_coord_rhand = np.amax(openpose_rhand[openpose_rhand_valid], axis=0)
            rcrop_center2d = 0.5 * (min_coord_rhand + max_coord_rhand)
            fit_size_rhand = np.amax(np.maximum(max_coord_rhand - rcrop_center2d, rcrop_center2d - min_coord_rhand))
            crop_size_best_r = 2 * fit_size_rhand * hand_zoom
        else:
            rcrop_center2d = np.array([-1000., -1000.])
            fit_size_rhand = 100
            crop_size_best_r = 2 * fit_size_rhand * hand_zoom
        if openpose_lhand_valid.any():
            min_coord_lhand = np.amin(openpose_lhand[openpose_lhand_valid], axis=0)
            max_coord_lhand = np.amax(openpose_lhand[openpose_lhand_valid], axis=0)
            lcrop_center2d = 0.5 * (min_coord_lhand + max_coord_lhand)
            fit_size_lhand = np.amax(np.maximum(max_coord_lhand - lcrop_center2d, lcrop_center2d - min_coord_lhand))
            crop_size_best_l = 2 * fit_size_lhand * hand_zoom
        else:
            lcrop_center2d = np.array([-1000., -1000.])
            fit_size_lhand = 100
            crop_size_best_l = 2 * fit_size_lhand * hand_zoom
        if not TRACK_HAND:
            rscale2d = float(s[1]) / crop_size_best_r
            lscale2d = float(s[1]) / crop_size_best_l
    else:
        # flag, boxes = tracker.update(image_v)
        gray_prev_image = cv2.cvtColor(prev_image_v, cv2.COLOR_RGB2GRAY)
        gray_current_image = cv2.cvtColor(image_v, cv2.COLOR_RGB2GRAY)
        l_lk_params = {'winSize': (int(2 * lhand_track_size), int(2 * lhand_track_size)), 'maxLevel': 3}
        lp, lstatus, error = cv2.calcOpticalFlowPyrLK(gray_prev_image, gray_current_image, lcenter.reshape(1, 2), None, **l_lk_params)
        lp_2, lstatus_2, error_2 = cv2.calcOpticalFlowPyrLK(gray_current_image, gray_prev_image, lp, None, **l_lk_params)
        if nl.norm(lp_2[0] - lcenter) > BACK_TRACK_THRESH or error[0] > 15:
            print ('LK left hand failed.')
            lstatus[0] = 0
        r_lk_params = {'winSize': (int(2 * rhand_track_size), int(2 * rhand_track_size)), 'maxLevel': 3}
        rp, rstatus, error = cv2.calcOpticalFlowPyrLK(gray_prev_image, gray_current_image, rcenter.reshape(1, 2), None, **r_lk_params)
        rp_2, rstatus_2, error_2 = cv2.calcOpticalFlowPyrLK(gray_current_image, gray_prev_image, rp, None, **r_lk_params)
        if nl.norm(rp_2[0] - rcenter) > BACK_TRACK_THRESH or error[0] > 15:
            print ('LK right hand failed.')
            rstatus[0] = 0

        lcrop_center2d_last = lcrop_center2d
        rcrop_center2d_last = rcrop_center2d
        if lstatus[0]:
            lcrop_center2d = lp[0]
        elif openpose_lhand_valid.any():
            min_coord_lhand = np.amin(openpose_lhand[openpose_lhand_valid], axis=0)
            max_coord_lhand = np.amax(openpose_lhand[openpose_lhand_valid], axis=0)
            lcrop_center2d = 0.5 * (min_coord_lhand + max_coord_lhand)
        elif lwrist_valid and lwrist_valid_last:
            lcrop_center2d = lcrop_center2d_last + lwrist - lwrist_last
        if rstatus[0]:
            rcrop_center2d = rp[0]
        elif openpose_rhand_valid.any():
            min_coord_rhand = np.amin(openpose_rhand[openpose_rhand_valid], axis=0)
            max_coord_rhand = np.amax(openpose_rhand[openpose_rhand_valid], axis=0)
            rcrop_center2d = 0.5 * (min_coord_rhand + max_coord_rhand)
        elif rwrist_valid and rwrist_valid_last:
            rcrop_center2d = rcrop_center2d_last + rwrist - rwrist_last
            # rcrop_center2d = rcenter + rwrist - rwrist_last

        # chest the distance between wrist & hand bbox, and the velocity of wrist & hand bbox
        # Also, if valid keypoint is too few, then don't trust the tracking result.
        if np.sum(lhand_valid) < 5 or \
            lwrist_valid and nl.norm(lwrist - lcrop_center2d) / lhand_track_size > 2 or \
            (lwrist_valid and lwrist_valid_last and
             nl.norm(lwrist - lwrist_last - lcrop_center2d + lcrop_center2d_last) / lhand_track_size > 1):
            print ('tracking left hand lost, starting from openpose')
            if openpose_lhand_valid.any():
                min_coord_lhand = np.amin(openpose_lhand[openpose_lhand_valid], axis=0)
                max_coord_lhand = np.amax(openpose_lhand[openpose_lhand_valid], axis=0)
                lcrop_center2d = 0.5 * (min_coord_lhand + max_coord_lhand)
            elif lwrist_valid:
                lcrop_center2d = lwrist
            elif lwrist_valid_last:
                lcrop_center2d = lwrist_last
            else:
                # If Openpose not available and no wrist is available, then do not update the cropping center
                lcrop_center2d = lcrop_center2d_last
        if np.sum(rhand_valid) < 5 or \
            rwrist_valid and nl.norm(rwrist - rcrop_center2d) / rhand_track_size > 2 or \
            (rwrist_valid and rwrist_valid_last and
             nl.norm(rwrist - rwrist_last - rcrop_center2d + rcrop_center2d_last) / rhand_track_size > 1):
            print ('tracking right hand lost, starting from openpose')
            if openpose_rhand_valid.any():
                min_coord_rhand = np.amin(openpose_rhand[openpose_rhand_valid], axis=0)
                max_coord_rhand = np.amax(openpose_rhand[openpose_rhand_valid], axis=0)
                rcrop_center2d = 0.5 * (min_coord_rhand + max_coord_rhand)
            elif rwrist_valid:
                rcrop_center2d = rwrist
            elif rwrist_valid_last:
                rcrop_center2d = rwrist_last
            else:
                # If Openpose not available and no wrist is available, then do not update the cropping center
                rcrop_center2d = rcrop_center2d_last

    rcrop_center2d_origin[i] = rcrop_center2d
    val_dict['rcrop_center2d'] = rcrop_center2d
    rscale2d_origin[i] = rscale2d
    val_dict['rscale2d'] = rscale2d
    rH = np.array([[rscale2d, 0, s[2] / 2 - rscale2d * rcrop_center2d[0]], [0, rscale2d, s[1] / 2 - rscale2d * rcrop_center2d[1]]], dtype=np.float32)
    rimage_crop_v = cv2.warpAffine(image_v, rH, (s[2], s[1]), flags=cv2.INTER_LANCZOS4)
    rimage_crop_v_feed = np.expand_dims((rimage_crop_v / 255 - 0.5), axis=0)

    lcrop_center2d_origin[i] = lcrop_center2d
    val_dict['lcrop_center2d'] = lcrop_center2d
    lscale2d_origin[i] = lscale2d
    val_dict['lscale2d'] = lscale2d
    lH = np.array([[lscale2d, 0, s[2] / 2 - lscale2d * lcrop_center2d[0]], [0, lscale2d, s[1] / 2 - lscale2d * lcrop_center2d[1]]], dtype=np.float32)
    limage_crop_v = cv2.warpAffine(image_v, lH, (s[2], s[1]), flags=cv2.INTER_LANCZOS4)
    limage_crop_v_feed = np.expand_dims((limage_crop_v / 255 - 0.5), axis=0)

    lheatmap_2d, lPAF, rheatmap_2d, rPAF = \
        [np.squeeze(_) for _ in
         sess.run([data['lheatmap_2d'], data['lPAF'], data['rheatmap_2d'], data['rPAF']],
                  feed_dict={data['limage_crop']: limage_crop_v_feed, data['rimage_crop']: rimage_crop_v_feed})]
    val_dict['rheatmap_2d'] = rheatmap_2d
    val_dict['rPAF'] = rPAF
    val_dict['lheatmap_2d'] = lheatmap_2d
    val_dict['lPAF'] = lPAF

    lhand2d_pred_v, lscore = utils.PAF.detect_keypoints2d_PAF(val_dict['lheatmap_2d'], val_dict['lPAF'], objtype=1)
    rhand2d_pred_v, rscore = utils.PAF.detect_keypoints2d_PAF(val_dict['rheatmap_2d'], val_dict['rPAF'], objtype=1)
    lhand2d_pred_v = lhand2d_pred_v[:21, :]
    rhand2d_pred_v = rhand2d_pred_v[:21, :]
    lhand_valid = lscore > 0.20  # false means that openpose fails to give the correct bounding box for hands
    rhand_valid = rscore > 0.20

    lhand2d_pred_v[np.logical_not(lhand_valid)] = 0  # must do this, otherwise PAF_to_3D error
    rhand2d_pred_v[np.logical_not(rhand_valid)] = 0  # must do this, otherwise PAF_to_3D error

    # check whether the keypoint is out of image
    lorigin = (lhand2d_pred_v - 184) / val_dict['lscale2d'] + val_dict['lcrop_center2d']
    lout = (lorigin[:, 0] < 0) + (lorigin[:, 1] < 0) + (lorigin[:, 0] >= args.width) + (lorigin[:, 1] >= args.height)
    lhand2d_pred_v[lout] = 0.0
    lhand_valid[lout] = False
    rorigin = (rhand2d_pred_v - 184) / val_dict['rscale2d'] + val_dict['rcrop_center2d']
    rout = (rorigin[:, 0] < 0) + (rorigin[:, 1] < 0) + (rorigin[:, 0] >= args.width) + (rorigin[:, 1] >= args.height)
    rhand2d_pred_v[rout] = 0.0
    rhand_valid[rout] = False

    if args.freeze:
        # freeze the torso
        body2d_pred_v[8:14] = 0
        body_valid[8:14] = 0
        body2d_pred_v[19] = 0
        body_valid[19] = 0

    # rescale 2D detection back to the original image
    body_2d = {'uv_local': body2d_pred_v, 'scale2d': val_dict['bscale2d'], 'crop_center2d': val_dict['bcrop_center2d'], 'valid': body_valid}
    lhand_2d = {'uv_local': lhand2d_pred_v, 'scale2d': val_dict['lscale2d'], 'crop_center2d': val_dict['lcrop_center2d'], 'valid': lhand_valid}
    rhand_2d = {'uv_local': rhand2d_pred_v, 'scale2d': val_dict['rscale2d'], 'crop_center2d': val_dict['rcrop_center2d'], 'valid': rhand_valid}

    total_keypoints_2d = utils.keypoint_conversion.assemble_total_2d(body_2d, lhand_2d, rhand_2d)  # put back to original image size, and change the keypoint order
    openpose_face = val_dict['openpose_face']
    openpose_face[:, 0] *= (val_dict['openpose_face_score'] > 0.5)  # Face must have a high threshold in case of occlusion.
    openpose_face[:, 1] *= (val_dict['openpose_face_score'] > 0.5)
    openpose_foot = val_dict['openpose_foot']
    openpose_foot[:, 0] *= (val_dict['openpose_foot_score'] > 0.05)
    openpose_foot[:, 1] *= (val_dict['openpose_foot_score'] > 0.05)
    total_keypoints_2d = np.concatenate([total_keypoints_2d, openpose_face, openpose_foot], axis=0)  # has dimension 20 + 21 + 21 + 70 + 6

    # extract PAF vectors from network prediction
    body3d_pred_v, _ = utils.PAF.PAF_to_3D(body2d_pred_v, val_dict['bPAF'], objtype=0)  # vec3ds has 18 rows, excluding shoulder to ear connection, only 14 used for fitting
    vec3ds = utils.PAF.collect_PAF_vec(body2d_pred_v, val_dict['bPAF'], objtype=0)  # vec3ds has 18 rows, excluding shoulder to ear connection, only 14 used for fitting
    lhand3d_pred_v, _ = utils.PAF.PAF_to_3D(lhand2d_pred_v, val_dict['lPAF'], objtype=1)
    lvec3ds = utils.PAF.collect_PAF_vec(lhand2d_pred_v, val_dict['lPAF'], objtype=1)
    rhand3d_pred_v, _ = utils.PAF.PAF_to_3D(rhand2d_pred_v, val_dict['rPAF'], objtype=1)
    rvec3ds = utils.PAF.collect_PAF_vec(rhand2d_pred_v, val_dict['rPAF'], objtype=1)
    body3d_pred_v[np.logical_not(body_valid)] = 0
    lhand3d_pred_v[np.logical_not(lhand_valid)] = 0
    rhand3d_pred_v[np.logical_not(rhand_valid)] = 0
    bPAF_valid = utils.PAF.getValidPAFNumpy(body_valid, 0)  # A PAF is valid only if both end points are valid.
    lPAF_valid = utils.PAF.getValidPAFNumpy(lhand_valid, 1)
    rPAF_valid = utils.PAF.getValidPAFNumpy(rhand_valid, 1)
    vec3ds[np.logical_not(bPAF_valid[BODY_PAF_SELECT_INDEX])] = 0
    lvec3ds[np.logical_not(lPAF_valid)] = 0
    rvec3ds[np.logical_not(rPAF_valid)] = 0

    if args.freeze:
        total_keypoints_2d[-6:] = 0
        vec3ds[:6] = np.array([0., 1., 0.])
        vec3ds[-3:] = 0

    # all limbs plus neck -> nose, neck -> headtop, 3 connections with chest, (additional 6 connection), left hand, right hand (14 + 3 + 6 + 20 + 20)
    PAF_vec = np.concatenate((vec3ds[:13, :], vec3ds[-4:, :], np.zeros([6, 3]), lvec3ds, rvec3ds), axis=0)
    with open(os.path.join(args.path, 'net_output', '{:012d}.txt'.format(frame_index)), 'w') as f:
        f.write('2D keypoints:\n')
        for kp in total_keypoints_2d:
            f.write('{} {}\n'.format(kp[0], kp[1]))
        f.write('PAF:\n')
        for vec in PAF_vec:
            f.write('{} {} {}\n'.format(vec[0], vec[1], vec[2]))
        f.write('{}\n'.format(float(np.sum(lscore) > 10)))
        f.write('{}\n'.format(float(np.sum(rscore) > 10)))
        if (np.sum(lscore) < 10):
            print('Left hand blurry.')
        if (np.sum(rscore) < 10):
            print('Right hand blurry.')

    if lhand_valid.any():
        lcenter = 0.5 * (np.amin(lorigin[lhand_valid], axis=0) + np.amax(lorigin[lhand_valid], axis=0)).astype(np.float32)  # detection center
    else:
        lcenter = lcrop_center2d.astype(np.float32)
    if rhand_valid.any():
        rcenter = 0.5 * (np.amin(rorigin[rhand_valid], axis=0) + np.amax(rorigin[rhand_valid], axis=0)).astype(np.float32)  # detection center
    else:
        rcenter = rcrop_center2d.astype(np.float32)
    lhand_track_size = fit_size_lhand * lscale2d_origin[start_i] / lscale2d
    rhand_track_size = fit_size_rhand * rscale2d_origin[start_i] / rscale2d
    prev_image_v = image_v

    if args.visualize:
        nc = 3
        nr = 4
        fig = plt.figure(1)
        ax1 = fig.add_subplot(nc, nr, 1)
        plt.imshow(bimage_crop_v)
        utils.general.plot2d(ax1, body2d_pred_v, valid_idx=body_valid, type_str=utils.general.type_strs[0], color=np.array([0.0, 0.0, 1.0]))

        ax2 = fig.add_subplot(nc, nr, 2)
        ax2.imshow(limage_crop_v)
        utils.general.plot2d(ax2, lhand2d_pred_v, type_str=utils.general.type_strs[1], color=np.array([0.0, 0.0, 1.0]))

        ax3 = fig.add_subplot(nc, nr, 3)
        ax3.imshow(rimage_crop_v)
        utils.general.plot2d(ax3, rhand2d_pred_v, type_str=utils.general.type_strs[1], color=np.array([0.0, 0.0, 1.0]))

        ax4 = fig.add_subplot(nc, nr, 4)
        bPAF_xy, bPAF_z = utils.PAF.plot_all_PAF(val_dict['bPAF'], 3)
        ax4.imshow(bPAF_xy)

        ax5 = fig.add_subplot(nc, nr, 5)
        ax5.imshow(bPAF_z)

        ax6 = fig.add_subplot(nc, nr, 6)
        plt.imshow(image_v)
        utils.general.plot2d(ax6, total_keypoints_2d, type_str='total', s=5)

        ax7 = fig.add_subplot(nc, nr, 7, projection='3d')
        utils.general.plot3d(ax7, body3d_pred_v, valid_idx=body_valid, type_str=utils.general.type_strs[0], color=np.array([0.0, 0.0, 1.0]))
        ax7.set_xlabel('X Label')
        ax7.set_ylabel('Y Label')
        ax7.set_zlabel('Z Label')
        plt.axis('equal')

        ax8 = fig.add_subplot(nc, nr, 8, projection='3d')
        utils.general.plot3d(ax8, lhand3d_pred_v, type_str=utils.general.type_strs[1], color=np.array([0.0, 0.0, 1.0]))
        ax8.set_xlabel('X Label')
        ax8.set_ylabel('Y Label')
        ax8.set_zlabel('Z Label')
        plt.axis('equal')

        ax9 = fig.add_subplot(nc, nr, 9, projection='3d')
        utils.general.plot3d(ax9, rhand3d_pred_v, type_str=utils.general.type_strs[1], color=np.array([0.0, 0.0, 1.0]))
        ax9.set_xlabel('X Label')
        ax9.set_ylabel('Y Label')
        ax9.set_zlabel('Z Label')
        plt.axis('equal')

        plt.show()

    if args.save_image:
        utils.general.plot2d_cv2(bimage_crop_v, body2d_pred_v, s=5, valid_idx=body_valid, use_color=False)
        assert cv2.imwrite(os.path.join(args.path, 'body_2d', '{:04d}.png'.format(i)), bimage_crop_v[:, :, ::-1])
        bPAF_xy, bPAF_z = utils.PAF.plot_all_PAF(val_dict['bPAF'], 3)
        k = 1. / val_dict['bscale2d']
        tx, ty = (val_dict['bcrop_center2d'] - 184 * k).astype(int)
        M = np.array([[k, 0., tx], [0., k, ty]], dtype=np.float32)
        resized_PAF_xy = cv2.warpAffine(bPAF_xy, M, (1920, 1080))[:args.height, :args.width, :]
        resized_PAF_z = cv2.warpAffine(bPAF_z, M, (1920, 1080))[:args.height, :args.width, :]
        assert cv2.imwrite(os.path.join(args.path, 'paf_xy_body', '{:04d}.png'.format(frame_index)), 255 - resized_PAF_xy[:, :, ::-1])
        assert cv2.imwrite(os.path.join(args.path, 'paf_z_body', '{:04d}.png'.format(frame_index)), 255 - resized_PAF_z[:, :, ::-1])

        utils.general.plot2d_cv2(limage_crop_v, lhand2d_pred_v, type_str='hand', s=5, use_color=True)
        lPAF_xy, lPAF_z = utils.PAF.plot_all_PAF(val_dict['lPAF'], 3)
        assert cv2.imwrite(os.path.join(args.path, 'lhand_2d', '{:04d}.png'.format(frame_index)), limage_crop_v[:, :, ::-1])
        assert cv2.imwrite(os.path.join(args.path, 'paf_xy_lhand', '{:04d}.png'.format(frame_index)), 255 - lPAF_xy[:, :, ::-1])
        assert cv2.imwrite(os.path.join(args.path, 'paf_z_lhand', '{:04d}.png'.format(frame_index)), 255 - lPAF_z[:, :, ::-1])

        utils.general.plot2d_cv2(rimage_crop_v, rhand2d_pred_v, type_str='hand', s=5, use_color=True)
        rPAF_xy, rPAF_z = utils.PAF.plot_all_PAF(val_dict['rPAF'], 3)
        assert cv2.imwrite(os.path.join(args.path, 'rhand_2d', '{:04d}.png'.format(frame_index)), rimage_crop_v[:, :, ::-1])
        assert cv2.imwrite(os.path.join(args.path, 'paf_xy_rhand', '{:04d}.png'.format(frame_index)), 255 - rPAF_xy[:, :, ::-1])
        assert cv2.imwrite(os.path.join(args.path, 'paf_z_rhand', '{:04d}.png'.format(frame_index)), 255 - rPAF_z[:, :, ::-1])
