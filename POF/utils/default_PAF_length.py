from utils.AdamModel import AdamModel
from utils.PAF import PAFConnection
import tensorflow as tf
import numpy as np
import json

if __name__ == '__main__':
    adam = AdamModel()
    adam_joints = adam.reconstruct()
    sess = tf.Session()
    V_vec, joints_v = sess.run([adam.mean_shape, adam_joints])
    sess.close()
    joints_v = joints_v.reshape(adam.num_joints, 3)
    V = V_vec.reshape(adam.num_vertices, 3)

    coords3d = np.zeros([19, 3], dtype=np.float64)
    coords3d[1] = joints_v[12]
    coords3d[2] = joints_v[17]
    coords3d[3] = joints_v[19]
    coords3d[4] = joints_v[21]
    coords3d[5] = joints_v[16]
    coords3d[6] = joints_v[18]
    coords3d[7] = joints_v[20]
    coords3d[8] = joints_v[2]
    coords3d[9] = joints_v[5]
    coords3d[10] = joints_v[8]
    coords3d[11] = joints_v[1]
    coords3d[12] = joints_v[4]
    coords3d[13] = joints_v[7]
    coords3d[0] = V[8130]
    coords3d[16] = V[10088]
    coords3d[17] = V[6970]
    coords3d[18] = V[1372]
    coords3d[14] = V[9707]
    coords3d[15] = V[2058]

    PAF_lengths = [[], []]
    for conn in PAFConnection[0]:
        vector = coords3d[conn[1]] - coords3d[conn[0]]
        length = np.sqrt(vector.dot(vector))
        PAF_lengths[0].append(length)

    coords3d_hand = np.zeros([21, 3], dtype=np.float64)
    coords3d_hand[0] = joints_v[20]
    coords3d_hand[1] = joints_v[25]
    coords3d_hand[2] = joints_v[24]
    coords3d_hand[3] = joints_v[23]
    coords3d_hand[4] = joints_v[22]

    coords3d_hand[5] = joints_v[29]
    coords3d_hand[6] = joints_v[28]
    coords3d_hand[7] = joints_v[27]
    coords3d_hand[8] = joints_v[26]

    coords3d_hand[9] = joints_v[33]
    coords3d_hand[10] = joints_v[32]
    coords3d_hand[11] = joints_v[31]
    coords3d_hand[12] = joints_v[30]

    coords3d_hand[13] = joints_v[37]
    coords3d_hand[14] = joints_v[36]
    coords3d_hand[15] = joints_v[35]
    coords3d_hand[16] = joints_v[34]

    coords3d_hand[17] = joints_v[41]
    coords3d_hand[18] = joints_v[40]
    coords3d_hand[19] = joints_v[39]
    coords3d_hand[20] = joints_v[38]

    for conn in PAFConnection[1]:
        vector = coords3d_hand[conn[1]] - coords3d_hand[conn[0]]
        length = np.sqrt(vector.dot(vector))
        PAF_lengths[1].append(length)

    with open('utils/default_PAF_lengths.json', 'w') as f:
        json.dump(PAF_lengths, f)
