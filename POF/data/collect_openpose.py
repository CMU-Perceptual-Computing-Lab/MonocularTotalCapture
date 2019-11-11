import os
import numpy as np
import numpy.linalg as nl
import json
import pickle

map_body25_to_body19 = list(range(8)) + list(range(9, 25))  # total of 24

seqName = 'Dexter_Grasp2'
# root = '/home/donglaix/Documents/Experiments/{}'.format(seqName)
root = '/media/posefs1b/Users/donglaix/siggasia018/{}/'.format(seqName)

calib_file = os.path.join(root, 'calib.json')
with open(calib_file) as f:
    calib_data = json.load(f)

start = 0
end = 648
frameRange = range(start, end)
person_idx = -1
# -1 for most obvious person, -2 for second obvious person

bs = []
ls = []
rs = []
fs = []
img_dirs = []

for i in frameRange:
    # img_file = os.path.join('openpose_image', '{}_{:012d}.jpg'.format(seqName, i)) if os.path.exists(os.path.join(root, 'openpose_image', '{}_{:012d}.jpg'.format(seqName, i))) \
    #     else os.path.join('openpose_image', '{}_{:012d}.png'.format(seqName, i))  # Openpose run on images
    img_file = os.path.join('openpose_image', '{}_{:012d}_rendered.png'.format(seqName, i))  # Openpose run on video
    assert os.path.exists(os.path.join(root, img_file))
    annot_2d = os.path.join(root, 'openpose_result', '{}_{:012d}_keypoints.json'.format(seqName, i))
    assert os.path.exists(annot_2d)
    with open(annot_2d) as f:
        data = json.load(f)
    scores = []
    areas = []
    for ip in range(len(data["people"])):
        joint2d = np.array(data["people"][ip]["pose_keypoints_2d"]).reshape(-1, 3)
        left_hand2d = np.array(data["people"][ip]["hand_left_keypoints_2d"]).reshape(-1, 3)
        right_hand2d = np.array(data["people"][ip]["hand_right_keypoints_2d"]).reshape(-1, 3)
        face2d = np.array(data["people"][ip]["face_keypoints_2d"]).reshape(-1, 3)
        score = np.sum(joint2d[:, 2]) + np.sum(left_hand2d[:, 2]) + np.sum(right_hand2d[:, 2]) + np.sum(face2d[:, 2])
        scores.append(score)
        joint_valid = (joint2d[:, 0] > 0.0) * (joint2d[:, 1] > 0.0)
        joint_nonzero = joint2d[joint_valid, :][:, :2]
        mx, my = joint_nonzero.min(axis=0)
        Mx, My = joint_nonzero.max(axis=0)
        areas.append((Mx - mx) * (My - my))
    scores = np.array(scores)
    areas = np.array(areas)

    idx = np.argsort(scores)
    # idx = np.argsort(areas)
    ip = idx[person_idx]

    joint2d = np.array(data["people"][ip]["pose_keypoints_2d"]).reshape(-1, 3)
    left_hand2d = np.array(data["people"][ip]["hand_left_keypoints_2d"]).reshape(-1, 3)
    right_hand2d = np.array(data["people"][ip]["hand_right_keypoints_2d"]).reshape(-1, 3)
    face2d = np.array(data["people"][ip]["face_keypoints_2d"]).reshape(-1, 3)
    final_body = joint2d[map_body25_to_body19]
    final_left = left_hand2d
    final_right = right_hand2d
    final_face = face2d

    bs.append(final_body)
    fs.append(final_face)
    ls.append(final_left)
    rs.append(final_right)
    img_dirs.append(img_file)

img_dirs = np.array(img_dirs)
bs = np.array(bs)
ls = np.array(ls)
rs = np.array(rs)
fs = np.array(fs)

print((len(ls), len(rs), len(fs), len(bs), len(img_dirs)))

with open('{}.pkl'.format(seqName), 'wb') as f:
    pickle.dump((bs, ls, rs, fs, img_dirs, calib_data), f)
