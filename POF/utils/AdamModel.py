import tensorflow as tf
import json
import numpy as np


class AdamModel(object):

    num_shape_coeff = 30
    num_vertices = 18540
    num_joints = 62

    def __init__(self):
        # read in model file
        model_file = 'utils/adam_v1_plus2.json'
        with open(model_file) as f:
            model_data = json.load(f)

        pca_file = 'utils/adam_blendshapes_348_delta_norm.json'
        with open(pca_file) as f:
            pca_data = json.load(f)

        with tf.variable_scope("AdamModel"):
            self.mean_shape = tf.constant(np.array(pca_data['mu']), shape=(self.num_vertices * 3,), name='mean_shape', dtype=tf.float32)
            self.shape_basis = tf.constant(np.array(pca_data['Uw1']), name='shape_basis', dtype=tf.float32)

            J_reg_sparse = model_data['adam_J_regressor_big']
            J_reg_size = np.array(J_reg_sparse[0], dtype=np.int32)[:2]
            J_reg = np.array(J_reg_sparse[1:], dtype=np.float32)
            J_reg_indices = J_reg[:, :2].astype(np.int32)
            J_reg_vals = J_reg[:, 2]
            self.J_reg = tf.sparse_reorder(tf.SparseTensor(J_reg_indices, J_reg_vals, J_reg_size))
            self.J_reg_dense = tf.sparse_tensor_to_dense(self.J_reg)

            # parental relationship (for forward_kinametics)
            kintree_table = np.array(model_data['kintree_table'], dtype=np.int32)
            id_to_col = np.zeros((self.num_joints), dtype=np.int32)
            self.m_parent = np.zeros((self.num_joints), dtype=np.int32)  # !: This is numpy array.

            for i in range(kintree_table.shape[1]):
                id_to_col[kintree_table[1, i]] = i
            for i in range(1, kintree_table.shape[1]):
                self.m_parent[i] = id_to_col[kintree_table[0, i]]

    def reconstruct(self, pose=None, coeff=None, trans=None):
        with tf.variable_scope("AdamModel"):
            if pose is None and coeff is None:
                batch_size = 1
            else:
                if pose is not None:
                    batch_size = pose.get_shape().as_list()[0]
                else:
                    batch_size = coeff.get_shape().as_list()[0]

            if coeff is None:
                coeff = tf.zeros((batch_size, self.num_shape_coeff), dtype=tf.float32)
            assert len(coeff.get_shape().as_list()) == 2  # [batch_size, shape_coeff]
            batch_size = coeff.get_shape().as_list()[0]
            V = self.mean_shape + tf.matmul(coeff, self.shape_basis, transpose_b=True)  # mean + shape_basis * shape_coeff
            # mat_V = tf.reshape(V, [self.num_vertices, 3])

            # J0 = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_reg, V))
            J0 = tf.matmul(V, self.J_reg_dense, transpose_b=True)
            mat_J0 = tf.reshape(J0, [batch_size, -1, 3])

            if pose is None:
                pose = tf.zeros((batch_size, 3 * self.num_joints), dtype=tf.float32)  # note different size with coeff
            assert len(pose.get_shape().as_list()) == 2   # [batch_size, 3 * num_joints]

            Js = []
            for i in range(batch_size):
                mat_J = self.forward_kinametics(mat_J0[i, :, :], pose[i, :])
                if trans is not None:  # [batch_size, 3]
                    assert len(trans.get_shape().as_list()) == 2
                    mat_J = mat_J + trans[i, :]
                J = tf.reshape(mat_J, [-1])
                Js.append(J)
            Js = tf.stack(Js, axis=0)

        return Js

    def forward_kinametics(self, J0, pose):
        with tf.variable_scope("forward_kinametics"):
            Rs = []  # transformation matrix
            ts = []

            R0 = self.AngleAxisToRotationMatrix(pose[:3])
            t0 = tf.transpose(J0[0:1, :])

            Rs.append(R0)
            ts.append(t0)

            for idj in range(1, self.num_joints):
                ipar = self.m_parent[idj]
                if idj in (10, 11):  # foot ends
                    angles = tf.zeros((3,), dtype=pose.dtype)
                elif idj in (7, 8):  # foot ankle
                    angles = tf.concat([pose[idj * 3:(idj + 1) * 3 - 1], tf.zeros([1, ], dtype=pose.dtype)], axis=0)
                elif idj in (24, 26, 27, 28, 31, 32, 35, 39, 40, 44, 47, 48, 51, 52, 55, 56, 59, 60):
                    angles = tf.concat([tf.zeros([2, ], dtype=pose.dtype), pose[idj * 3 + 2:(idj + 1) * 3]], axis=0)
                else:
                    angles = pose[idj * 3:(idj + 1) * 3]
                R = self.EulerAngleToRotationMatrix(angles)  # in ceres function, R is assumed to be row major, but in adam_reconstruct_euler, R is column major.

                R = tf.matmul(Rs[ipar], R)
                t = ts[ipar] + tf.matmul(Rs[ipar], tf.transpose(J0[idj:(idj + 1), :] - J0[ipar:(ipar + 1), :]))

                Rs.append(R)
                ts.append(t)

            for idj in range(self.num_joints):
                ts[idj] = ts[idj] - tf.matmul(Rs[idj], tf.transpose(J0[idj:(idj + 1), :]))
            J_out = []
            for idj in range(self.num_joints):
                J_out.append(tf.matmul(Rs[idj], tf.transpose(J0[idj:(idj + 1), :])) + ts[idj])  # original pose -> transformed pose (world coordinate)

            J_out = tf.transpose(tf.concat(J_out, axis=1))
            return J_out

    @staticmethod
    def AngleAxisToRotationMatrix(angle_axis):
        """ angle_axis is a 3d vector whose direction points to the rotation axis and whose norm is the angle (in radians) """
        with tf.variable_scope("AngleAxisToRotationMatrix"):
            theta = tf.norm(angle_axis)

            cos = tf.cos(theta)
            sin = tf.sin(theta)
            xyz = tf.divide(angle_axis, theta)

            x = xyz[0]
            y = xyz[1]
            z = xyz[2]

            # when theta > 0
            R00 = cos + x * x * (1. - cos)
            R10 = sin * z + x * y * (1. - cos)
            R20 = -sin * y + x * z * (1. - cos)
            Rcol0 = tf.stack([R00, R10, R20], axis=0)

            R01 = x * y * (1. - cos) - z * sin
            R11 = cos + y * y * (1. - cos)
            R21 = x * sin + y * z * (1. - cos)
            Rcol1 = tf.stack([R01, R11, R21], axis=0)

            R02 = y * sin + x * z * (1. - cos)
            R12 = -x * sin + y * z * (1. - cos)
            R22 = cos + z * z * (1. - cos)
            Rcol2 = tf.stack([R02, R12, R22], axis=0)

            R = tf.stack([Rcol0, Rcol1, Rcol2], axis=1)

            # when theta == 0
            R_00 = tf.ones([], dtype=angle_axis.dtype)
            R_10 = angle_axis[2]
            R_20 = -angle_axis[1]
            R_col0 = tf.stack([R_00, R_10, R_20], axis=0)

            R_01 = -angle_axis[2]
            R_11 = tf.ones([], dtype=angle_axis.dtype)
            R_21 = angle_axis[0]
            R_col1 = tf.stack([R_01, R_11, R_21], axis=0)

            R_02 = angle_axis[1]
            R_12 = -angle_axis[0]
            R_22 = tf.ones([], dtype=angle_axis.dtype)
            R_col2 = tf.stack([R_02, R_12, R_22], axis=0)

            R_ = tf.stack([R_col0, R_col1, R_col2], axis=1)

            return tf.cond(tf.greater(theta, 0), lambda: R, lambda: R_)

    @staticmethod
    def EulerAngleToRotationMatrix(euler_angle):
        """ This function computes the rotation matrix corresponding to Euler Angle (x, y, z) R_z * R_y * R_x (consistent with Ceres). (x, y, z) in degrees."""
        with tf.variable_scope("EulerAngleToRotationMatrix"):
            deg = euler_angle * np.pi / 180
            cos = tf.cos(deg)
            sin = tf.sin(deg)

            c3 = cos[0]
            c2 = cos[1]
            c1 = cos[2]

            s3 = sin[0]
            s2 = sin[1]
            s1 = sin[2]

            R00 = c1 * c2
            R10 = s1 * c2
            R20 = -s2
            Rcol0 = tf.stack([R00, R10, R20], axis=0)

            R01 = -s1 * c3 + c1 * s2 * s3
            R11 = c1 * c3 + s1 * s2 * s3
            R21 = c2 * s3
            Rcol1 = tf.stack([R01, R11, R21], axis=0)

            R02 = s1 * s3 + c1 * s2 * c3
            R12 = -c1 * s3 + s1 * s2 * c3
            R22 = c2 * c3
            Rcol2 = tf.stack([R02, R12, R22], axis=0)

            R = tf.stack([Rcol0, Rcol1, Rcol2], axis=1)
            return R


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    a = AdamModel()
    pose_np = np.zeros((2, 3 * 62,), dtype=np.float32)
    pose_np[0, 3 * 16 + 1] = -90.
    pose = tf.Variable(pose_np)
    J = a.reconstruct(pose=pose)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    JJ = sess.run(J)

    JJ = JJ.reshape(2, -1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(JJ[0, :, 0], JJ[0, :, 1], JJ[0, :, 2], color='red')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(JJ[1, :, 0], JJ[1, :, 1], JJ[1, :, 2], color='red')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.axis('equal')

    # from meshWrapper import meshWrapper
    # meshlib = meshWrapper("/home/donglaix/Documents/Experiments/hand_model/build/libPythonWrapper.so")
    # meshlib.load_totalmodel()
    # meshlib.reset_value()
    # meshlib.cpose[:] = pose_np.tolist()

    # ax = fig.add_subplot(222)
    # img1 = meshlib.total_visualize(cameraMode=False, target=False, first_render=False, position=0)
    # ax.imshow(img1)

    # ax = fig.add_subplot(223)
    # img2 = meshlib.total_visualize(cameraMode=False, target=False, first_render=False, position=1)
    # ax.imshow(img2)

    # ax = fig.add_subplot(224)
    # img3 = meshlib.total_visualize(cameraMode=False, target=False, first_render=False, position=2)
    # ax.imshow(img3)

    plt.show()
