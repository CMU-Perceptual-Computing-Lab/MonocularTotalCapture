import pickle
from scipy.io import loadmat
import os
import numpy as np

PATH_TO_DATASET = '/media/posefs0c/Users/donglaix/Experiments/StereoHandTracking/'
TEST_SEQS = ['B1Counting', 'B1Random']
TRAIN_SEQS = ['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random', 'B5Counting', 'B5Random', 'B6Counting', 'B6Random']

K = np.diag([822.79041, 822.79041, 1.0]).astype(np.float32)
K[0, 2] = 318.47345
K[1, 2] = 250.31296

base = 120.054
Rl = np.eye(3, dtype=np.float32)
Rr = np.eye(3, dtype=np.float32)
tl = np.zeros((3,), dtype=np.float32)
tr = np.array([-base, 0, 0], dtype=np.float32)

if __name__ == '__main__':
    assert os.path.isdir(PATH_TO_DATASET)
    # collect the testing sequences
    all_test_data = np.zeros((0, 21, 3), dtype=np.float32)
    for test_seq in TEST_SEQS:
        mat_path = os.path.join(PATH_TO_DATASET, 'labels', test_seq + '_BB.mat')
        mat_data = loadmat(mat_path)
        mat_data = np.transpose(mat_data['handPara'], (2, 1, 0))
        all_test_data = np.concatenate((all_test_data, mat_data), axis=0)

    all_train_data = np.zeros((0, 21, 3), dtype=np.float32)
    for train_seq in TRAIN_SEQS:
        mat_path = os.path.join(PATH_TO_DATASET, 'labels', train_seq + '_BB.mat')
        mat_data = loadmat(mat_path)
        mat_data = np.transpose(mat_data['handPara'], (2, 1, 0))
        all_train_data = np.concatenate((all_train_data, mat_data), axis=0)

    with open('stb_collected.pkl', 'wb') as f:
        pickle.dump((all_train_data, all_test_data), f)
