import ctypes
from PIL import Image, ImageOps
import numpy as np


class meshWrapper(object):
    def __init__(self, lib_file='./utils/libPythonWrapper.so'):
        self.lib = ctypes.cdll.LoadLibrary(lib_file)

        # extern "C" void load_totalmodel(char* obj_file, char* model_file, char* pca_file);
        self.lib.load_totalmodel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.load_totalmodel.restype = None

        self.obj_file = ctypes.create_string_buffer('./utils/mesh_nofeet.obj'.encode('ascii'))
        self.model_file = ctypes.create_string_buffer('./utils/adam_v1_plus2.json'.encode('ascii'))
        self.pca_file = ctypes.create_string_buffer('./utils/adam_blendshapes_348_delta_norm.json'.encode('ascii'))
        self.correspondence_file = ctypes.create_string_buffer('./utils/correspondences_nofeet.txt'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/adam_cocoplus_regressor.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/reg_human36_angjooOrder_ls.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/reg_human36_angjooOrder_nonneg.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/reg_combined_angjoo1.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/regressor_0n.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/reg_human36_angjooOrder_regressor2_nonneg.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/reg_human36_angjooOrder_regressor2_nonneg_root.json'.encode('ascii'))
        # self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/regressor_0n1.json'.encode('ascii'))
        self.cocoplus_regressor_file = ctypes.create_string_buffer('./utils/regressor_0n1_root.json'.encode('ascii'))

        # extern "C" void fit_total3d(double* targetJoint, double* pose, double* coeff, double* trans)
        self.lib.fit_total3d.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5
        self.lib.fit_total3d.restype = None
        self.lib.fit_total2d.argtypes = [ctypes.POINTER(ctypes.c_double)] * 6
        self.lib.fit_total2d.restype = None
        self.lib.fit_total3d2d.argtypes = [ctypes.POINTER(ctypes.c_double)] * 7
        self.lib.fit_total3d2d.restype = None

        # extern "C" void fit_PAF_vec(double* targetJoint2d, double* PAF_vec, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff)
        self.lib.fit_PAF_vec.argtypes = [ctypes.POINTER(ctypes.c_double)] * 8 + [ctypes.c_uint, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
        self.lib.fit_PAF_vec.restype = None

        # Eigen::Matrix<double, 62, 3, Eigen::RowMajor> m_adam_pose;  //62 ==TotalModel::NUM_JOINTS
        # Eigen::Matrix<double, 30, 1> m_adam_coeffs;         //30 ==TotalModel::NUM_SHAPE_COEFFICIENTS
        # Eigen::Vector3d m_adam_t;
        self.cpose = (ctypes.c_double * (62 * 3))()
        self.ccoeff = (ctypes.c_double * 30)()
        self.ctrans = (ctypes.c_double * 3)()
        self.cface_coeff = (ctypes.c_double * 200)()
        self.ctarget_array = (ctypes.c_double * ((62 + 70 + 6) * 3))()
        self.ctarget_array_2d = (ctypes.c_double * ((63 + 70 + 6) * 2))()
        self.cret_bytes = (ctypes.c_ubyte * (600 * 600 * 4))()
        self.cfull_bytes = (ctypes.c_ubyte * (1920 * 1080 * 4))()
        self.cortho_bytes = (ctypes.c_ubyte * (1920 * 1080 * 4))()
        self.PAF_array = (ctypes.c_double * (63 * 3))()
        self.out_joint = (ctypes.c_double * (65 * 3))()  # regressor 2: 19 (small coco regressor) + 20 (hand) + 20 (hand) + 6 (feet)
        self.calibK = (ctypes.c_double * 9)()

        # extern "C" void Total_visualize(GLubyte* ret_bytes, double* targetJoint, uint CameraMode, uint position, bool meshSolid, float scale, int vis_type)
        self.lib.Total_visualize.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_double),
                                             ctypes.c_uint, ctypes.c_uint, ctypes.c_bool, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
        self.lib.Total_visualize.restype = None
        self.lib.VisualizeSkeleton.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_double), ctypes.c_uint, ctypes.c_uint, ctypes.c_float]
        self.lib.VisualizeSkeleton.restype = None

        self.lib.init_renderer.argtypes = []
        self.lib.init_renderer.restype = None

        self.lib.reconstruct_adam.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4 + [ctypes.c_int]
        self.lib.reconstruct_adam.restype = None

        self.lib.reconstruct_adam_mesh.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4 + [ctypes.c_int, ctypes.c_bool]
        self.lib.reconstruct_adam_mesh.restype = None

        self.lib.fit_h36m_groundtruth.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5
        self.lib.fit_h36m_groundtruth.restype = None

        self.lib.adam_refit.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_uint]
        self.lib.adam_refit.restype = None

        self.lib.adam_sequence_init.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_uint]
        self.lib.adam_sequence_init.restype = None

        self.lib.adam_hsiu_fit_dome.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_bool]
        self.lib.adam_hsiu_fit_dome.restype = None

    def reset_value(self):
        self.ctrans[:] = [0.0, 0.0, 500.0]
        self.ccoeff[:] = [0.0] * 30
        self.cpose[:] = [0.0] * (62 * 3)
        self.cface_coeff[:] = [0.0] * 200

    def load_totalmodel(self):
        self.lib.load_totalmodel(self.obj_file, self.model_file, self.pca_file, self.correspondence_file, self.cocoplus_regressor_file)

    def fit_total3d(self, joint3d):
        assert joint3d.shape[1] == 3, joint3d.shape
        self.ctarget_array[:joint3d.size] = joint3d.reshape(-1).tolist()
        self.lib.fit_total3d(self.ctarget_array, self.cpose, self.ccoeff, self.ctrans, self.cface_coeff)

    def total_visualize(self, cameraMode=0, target=True, first_render=False, position=0, meshSolid=True, scale=1.0, vis_type=1, show_joint=True):
        if cameraMode == 0:
            read_buffer = self.cret_bytes
            read_size = (600, 600)
        elif cameraMode == 1:
            read_buffer = self.cfull_bytes
            read_size = (1920, 1080)
        else:
            assert cameraMode == 2
            read_buffer = self.cortho_bytes
            read_size = (1920, 1080)
        if first_render:
            self.lib.Total_visualize(read_buffer, self.ctarget_array if target else None, ctypes.c_uint(cameraMode),
                                     ctypes.c_uint(position), ctypes.c_bool(meshSolid), ctypes.c_float(scale), ctypes.c_int(vis_type),
                                     ctypes.c_bool(show_joint))
            read_buffer[:] = [0] * len(read_buffer[:])
        self.lib.Total_visualize(read_buffer, self.ctarget_array if target else None, ctypes.c_uint(cameraMode),
                                 ctypes.c_uint(position), ctypes.c_bool(meshSolid), ctypes.c_float(scale), ctypes.c_int(vis_type),
                                 ctypes.c_bool(show_joint))
        img = bytes(read_buffer[:read_size[0] * read_size[1] * 4])
        img = Image.frombytes("RGBA", read_size, img)
        img = ImageOps.flip(img)
        return img

    def fit_total2d(self, joint2d, K):
        assert joint2d.shape[1] == 2, joint2d.shape
        assert K.shape == (3, 3), K
        self.calibK[:] = K.reshape(-1).tolist()
        self.ctarget_array_2d[:] = joint2d.reshape(-1).tolist()
        self.lib.fit_total2d(self.ctarget_array_2d, self.calibK, self.cpose, self.ccoeff, self.ctrans, self.cface_coeff)

    def fit_total3d2d(self, joint3d, joint2d, K):
        assert joint3d.shape[1] == 3, joint3d.shape
        assert joint2d.shape[1] == 2, joint2d.shape
        assert K.shape == (3, 3), K
        self.ctarget_array[:joint3d.size] = joint3d.reshape(-1).tolist()
        self.ctarget_array_2d[:] = joint2d.reshape(-1).tolist()
        self.calibK[:] = K.reshape(-1).tolist()
        self.lib.fit_total3d2d(self.ctarget_array, self.ctarget_array_2d, self.calibK, self.cpose, self.ccoeff, self.ctrans, self.cface_coeff)

    def visualize_skeleton(self, joint3d, cameraMode=0, first_render=False, position=0, scale=1.0):
        if cameraMode == 0:
            read_buffer = self.cret_bytes
            read_size = (600, 600)
        elif cameraMode == 1:
            read_buffer = self.cfull_bytes
            read_size = (1920, 1080)
        else:
            assert cameraMode == 2
            read_buffer = self.cortho_bytes
            read_size = (1920, 1080)

        read_buffer[:] = [0] * len(read_buffer[:])
        assert joint3d.shape[1] == 3, joint3d.shape
        self.ctarget_array[:joint3d.size] = joint3d.reshape(-1).tolist()

        if first_render:
            self.lib.VisualizeSkeleton(read_buffer, self.ctarget_array, ctypes.c_uint(cameraMode), ctypes.c_uint(position), ctypes.c_float(scale))
        self.lib.VisualizeSkeleton(read_buffer, self.ctarget_array, ctypes.c_uint(cameraMode), ctypes.c_uint(position), ctypes.c_float(scale))
        img = bytes(read_buffer[:read_size[0] * read_size[1] * 4])
        img = Image.frombytes("RGBA", read_size, img)
        img = ImageOps.flip(img)
        return img

    def fit_PAF_vec(self, joint2d, PAF_vec, K, joint3d=None, regressor_type=0, quan=False, fitPAFfirst=False, fit_face_exp=False):
        assert joint2d.shape == (139, 2), joint2d.shape
        assert K.shape == (3, 3), K
        assert PAF_vec.shape[1] == 3, PAF_vec.shape
        assert PAF_vec.shape[0] == 63, PAF_vec.shape
        if joint3d is not None:
            assert joint3d.shape[1] == 3, joint3d.shape
            self.ctarget_array[:] = joint3d.reshape(-1).tolist()
        self.calibK[:] = K.reshape(-1).tolist()
        self.ctarget_array_2d[:] = [0.0] * len(self.ctarget_array_2d[:])
        self.ctarget_array_2d[:joint2d.shape[0] * 2] = joint2d.reshape(-1).tolist()
        self.PAF_array[:PAF_vec.size] = PAF_vec.reshape(-1).tolist()
        self.lib.fit_PAF_vec(self.ctarget_array_2d, self.PAF_array, self.calibK, self.cpose, self.ccoeff, self.ctrans, self.cface_coeff,
                             None if joint3d is None else self.ctarget_array, ctypes.c_uint(regressor_type),
                             ctypes.c_bool(quan), ctypes.c_bool(fitPAFfirst), ctypes.c_bool(fit_face_exp))

    def adam_refit(self, joint3d, regressor_type):
        assert joint3d.shape[1] == 3, joint3d.shape
        self.ctarget_array[:] = joint3d.reshape(-1).tolist()
        self.lib.adam_refit(self.cpose, self.ccoeff, self.ctrans, self.cface_coeff, self.ctarget_array, regressor_type)

    def adam_sequence_init(self, joint3d, regressor_type):
        assert joint3d.shape[1] == 3, joint3d.shape
        self.ctarget_array[:] = joint3d.reshape(-1).tolist()
        self.lib.adam_sequence_init(self.cpose, self.ccoeff, self.ctrans, self.cface_coeff, self.ctarget_array, regressor_type)

    def adam_hsiu_fit_dome(self, target_joint, freeze_shape=False):
        assert target_joint.shape == (20, 3)
        self.ctarget_array[:60] = target_joint.reshape(-1).tolist()
        self.lib.adam_hsiu_fit_dome(self.cpose, self.ccoeff, self.ctrans, self.cface_coeff, self.ctarget_array, freeze_shape)

    def refit_eval_h36m(self, regressor_type, prior_weight=1.0):
        # refit Adam using skeleton reconstructed from current params, update params with pose prior && AngleAxis
        self.lib.refit_eval_h36m(self.cpose, self.ccoeff, self.ctrans, ctypes.c_uint(regressor_type), ctypes.c_double(prior_weight))

    def fitSingleStage(self, joint2d, PAF_vec, K, regressor_type=0, fit_face_exp=False):
        assert joint2d.shape == (139, 2), joint2d.shape
        assert K.shape == (3, 3), K
        assert PAF_vec.shape[1] == 3, PAF_vec.shape
        assert PAF_vec.shape[0] == 63, PAF_vec.shape
        self.calibK[:] = K.reshape(-1).tolist()
        self.ctarget_array_2d[:] = [0.0] * len(self.ctarget_array_2d[:])
        self.ctarget_array_2d[:joint2d.shape[0] * 2] = joint2d.reshape(-1).tolist()
        self.PAF_array[:PAF_vec.size] = PAF_vec.reshape(-1).tolist()
        self.lib.fitSingleStage(self.ctarget_array_2d, self.PAF_array, self.calibK, self.cpose, self.ccoeff, self.ctrans, self.cface_coeff,
                                ctypes.c_uint(regressor_type), ctypes.c_bool(fit_face_exp))
