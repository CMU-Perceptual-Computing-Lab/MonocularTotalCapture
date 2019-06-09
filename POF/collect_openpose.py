import os
import numpy as np
import numpy.linalg as nl
import json
import pickle
import argparse

map_body25_to_body19 = list(range(8)) + list(range(9, 25))  # total of 24

parser = argparse.ArgumentParser()
parser.add_argument('--seqName', '-n', type=str)
parser.add_argument('--rootDir', '-r', type=str)
parser.add_argument('--count', '-c', type=int)
args = parser.parse_args()

seqName = args.seqName
root = args.rootDir

calib_file = os.path.join(root, 'calib.json')
with open(calib_file) as f:
    calib_data = json.load(f)

frameRange = range(1, args.count + 1)
person_idx = -1

bs = []
ls = []
rs = []
fs = []
img_dirs = []
frame_indices = []

for i in frameRange:
    img_file = os.path.join(root, "raw_image", '{}_{:08d}.png'.format(seqName, i))
    assert os.path.isfile(img_file)
    annot_2d = os.path.join(root, 'openpose_result', '{}_{:08d}_keypoints.json'.format(seqName, i))
    assert os.path.exists(annot_2d)
    with open(annot_2d) as f:
        data = json.load(f)

    # ideally there should be only one person
    assert len(data['people']) == 1
    ip = 0

    joint2d = np.array(data["people"][ip]["pose_keypoints_2d"]).reshape(-1, 3)
    left_hand2d = np.array(data["people"][ip]["hand_left_keypoints_2d"]).reshape(-1, 3)
    right_hand2d = np.array(data["people"][ip]["hand_right_keypoints_2d"]).reshape(-1, 3)
    face2d = np.array(data["people"][ip]["face_keypoints_2d"]).reshape(-1, 3)

    bs.append(joint2d[map_body25_to_body19])
    fs.append(face2d)
    ls.append(left_hand2d)
    rs.append(right_hand2d)
    img_dirs.append(img_file)
    frame_indices.append(i)

img_dirs = np.array(img_dirs)
bs = np.array(bs)
ls = np.array(ls)
rs = np.array(rs)
fs = np.array(fs)
frame_indices = np.array(frame_indices)

print('Openpose output collected: data dimension:')
print((len(ls), len(rs), len(fs), len(bs), len(img_dirs), len(frame_indices)))

with open('{}/{}.pkl'.format(root, seqName), 'wb') as f:
    pickle.dump((bs, ls, rs, fs, img_dirs, calib_data, frame_indices), f)
