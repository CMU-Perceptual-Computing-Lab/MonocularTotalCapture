# Don't use anaconda for this
import ctypes
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


class wrapper_hand_model(object):
    def __init__(self, lib_file='./utils/libPythonWrapper.so', model_file='./utils/hand2_l_all_uv.json'):
        self.lib = ctypes.cdll.LoadLibrary(lib_file)

        self.fit_hand3d = self.lib.fit_hand3d
        self.fit_hand3d.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool]
        self.fit_hand3d.restype = None

        self.Opengl_visualize = self.lib.Opengl_visualize
        self.Opengl_visualize.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_bool, ctypes.c_uint, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
        self.Opengl_visualize.restype = None

        self.fit_hand2d = self.lib.fit_hand2d
        self.fit_hand2d.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool,
                                    ctypes.c_double, ctypes.c_int]
        self.fit_hand2d.restype = None

        self.extract_fit_result = self.lib.extract_fit_result
        self.extract_fit_result.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool]
        self.extract_fit_result.restype = None

        self.lib.set_calibK.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.lib.set_calibK.restype = None

        self.cmodel_file = ctypes.create_string_buffer(model_file.encode('ascii'))
        self.ctarget_array = (ctypes.c_double * 63)()
        self.ctrans = (ctypes.c_double * 3)()
        self.ccoeff = (ctypes.c_double * 63)()
        self.cpose = (ctypes.c_double * 63)()
        self.cret_bytes = (ctypes.c_ubyte * (600 * 600 * 3))()
        self.ctarget2d_array = (ctypes.c_double * 42)()
        self.calibK = (ctypes.c_double * 9)()
        self.cret_bytes_cam = (ctypes.c_ubyte * (1080 * 1920 * 4))()

        self.PAF_array = (ctypes.c_double * (20 * 3))()

    def reset_value(self):
        self.ctrans[:] = [0.0, 0.0, 2.0]
        self.ccoeff[:] = [1.0 for _ in range(63)]
        self.cpose[:] = [0.0 for _ in range(63)]

    def fit3d(self, joint3d, regressor_type=0, euler=True):
        assert joint3d.shape == (21, 3)
        self.ctarget_array[:] = joint3d.reshape(-1).tolist()
        self.fit_hand3d(self.ctarget_array, self.cmodel_file, self.cpose, self.ccoeff, self.ctrans, regressor_type, euler)
        trans = np.array(self.ctrans[:])
        pose = np.array(self.cpose[:]).reshape(-1, 3)
        coeff = np.array(self.ccoeff[:]).reshape(-1, 3)
        return trans, pose, coeff

    def fit2d(self, joint2d, calibK, PAF, regressor_type=0, euler=True, prior_weight=100.0, mode=0):
        assert joint2d.shape == (21, 2) and calibK.shape == (3, 3)
        self.ctarget2d_array[:] = joint2d.reshape(-1).tolist()
        self.calibK[:] = calibK.reshape(-1).tolist()
        assert PAF.size == len(self.PAF_array[:])
        self.PAF_array[:] = PAF.reshape(-1).tolist()
        self.fit_hand2d(self.ctarget2d_array, self.calibK, self.PAF_array, self.cmodel_file, self.cpose, self.ccoeff, self.ctrans,
                        regressor_type, euler, prior_weight, mode)
        trans = np.array(self.ctrans[:])
        pose = np.array(self.cpose[:]).reshape(-1, 3)
        coeff = np.array(self.ccoeff[:]).reshape(-1, 3)
        return trans, pose, coeff

    def render(self, cameraMode=False, target=True, first_render=False, position=0, regressor_type=0, stay=False, euler=True):
        if cameraMode:
            read_buffer = self.cret_bytes_cam
        else:
            read_buffer = self.cret_bytes
        if target:
            if first_render:
                self.Opengl_visualize(self.cmodel_file, read_buffer, self.cpose, self.ccoeff, self.ctrans, self.ctarget_array, ctypes.c_bool(cameraMode), position, regressor_type, stay, euler)
            self.Opengl_visualize(self.cmodel_file, read_buffer, self.cpose, self.ccoeff, self.ctrans, self.ctarget_array, ctypes.c_bool(cameraMode), position, regressor_type, stay, euler)
        else:
            if first_render:
                self.Opengl_visualize(self.cmodel_file, read_buffer, self.cpose, self.ccoeff, self.ctrans, None, ctypes.c_bool(cameraMode), position, regressor_type, stay, euler)
            self.Opengl_visualize(self.cmodel_file, read_buffer, self.cpose, self.ccoeff, self.ctrans, None, ctypes.c_bool(cameraMode), position, regressor_type, stay, euler)
        img = bytes(read_buffer)
        if not cameraMode:
            img = Image.frombytes("RGBA", (600, 600), img)
        else:
            img = Image.frombytes("RGBA", (1920, 1080), img)
        img = ImageOps.flip(img)
        return img

    def set_calibK(self, K):
        self.calibK[:] = K.reshape(-1).tolist()
        self.lib.set_calibK(self.calibK)


if __name__ == '__main__':
    import numpy as np
    wrapper = wrapper_hand_model("/home/donglaix/Documents/Experiments/hand_model/build/libPythonWrapper.so")
    joint3d = np.array([-33.3889, -173.355, -36.0744, -35.0518, -173.959, -37.7108, -36.5972, -176.126, -40.7544, -37.4367, -178.032, -43.6272, -38.7743, -178.843, -45.5877, -36.4731, -180.718, -38.2183, -37.0009, -181.596, -42.4443, -37.4651, -181.437, -45.0006, -37.7732, -181.458, -47.0573, -34.2598, -180.606, -38.3926, -35.2143, -
                        180.671, -43.2699, -36.3031, -179.876, -45.6931, -37.1902, -179.438, -47.745, -32.0926, -179.69, -38.4972, -33.7518, -179.847, -42.8798, -34.9357, -179.212, -45.3947, -35.7699, -178.853, -47.3468, -30.3247, -178.334, -39.2571, -31.8778, -178.837, -42.4667, -33.003, -178.501, -44.2697, -33.8762, -178.325, -45.8248]).reshape(-1, 3)
    joint2d = np.array([1284.646, 254.091, 1296.991, 248.479, 1319.012, 231.635, 1339.5621, 217.027, 1354.4766, 209.81, 1300.0491, 200.093, 1330.055, 192.596, 1348.5556, 192.777, 1363.3952, 191.943, 1299.7998, 202.764, 1334.6115,
                        200.494, 1352.7438, 204.628, 1368.139, 206.547, 1299.2785, 210.884, 1330.8779, 207.547, 1349.5700, 210.478, 1364.09, 211.918, 1303.6187, 221.421, 1326.7478, 216.127, 1340.2151, 217.196, 1351.8205, 217.42]).reshape(-1, 2)
    K = np.array([[1633.34, 0, 942.256], [0, 1628.84, 557.344], [0, 0, 1]])
    wrapper.fit3d(joint3d)
    wrapper.fit2d(joint2d, K)
    img = wrapper.render(cameraMode=False, first_render=True)
    # print(trans)
    # print(pose)
    # print(coeff)
    plt.imshow(img)
    plt.show()
