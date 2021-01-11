// As a wrapper to Python
#include <iostream>
#include <KinematicModel.h>
#include <VisualizedData.h>
#include <FitToBody.h>
#include <handm.h>
#include "Renderer.h"
#include <assert.h>
#include <GL/freeglut.h>
#include "totalmodel.h"
#include "pose_to_transforms.h"
#include "ModelFitter.h"
// #include <FreeImage.h>
using namespace std;
#define SMPL_VIS_SCALING 100.0f

extern "C" void fit_hand3d(double* targetJoint, char* hand_model_file, double* pose, double* coeff, double* trans, int regressor_type=0, bool euler=true);
extern "C" void Opengl_visualize(char* hand_model_file, GLubyte* ret_bytes, double* pose, double* coeff, double* trans, double* targetJoint, bool CameraMode, uint position=0, int regressor_type=0, bool stay=false, bool euler=true);
extern "C" void fit_hand2d(double* targetJoint2d, double* calibK, double* PAF_array, char* hand_model_file, double* pose, double* coeff, double* trans, int regressor_type=0, bool euler=true, double prior_weight=100., int mode=0);
extern "C" void extract_fit_result(char* hand_model_file, double* pose, double* coeff, double* trans, double* resultJoint, int regressor_type=0, bool euler=true);
extern "C" void load_totalmodel(char* obj_file, char* model_file, char* pca_file, char* correspondence_file, char* cocoplus_regressor_file);
extern "C" void init_renderer();
extern "C" void fit_total3d(double* targetJoint, double* pose, double* coeff, double* trans, double* face_coeff);
extern "C" void fit_total2d(double* targetJoint2d, double* K, double* pose, double* coeff, double* trans, double* face_coeff);
extern "C" void fit_total3d2d(double* targetJoint, double* targetJoint2d, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff);
extern "C" void Total_visualize(GLubyte* ret_bytes, double* targetJoint, uint CameraMode, uint position=0, bool meshSolid=true, float scale=1.0, int vis_type=1, bool show_joint=true);
extern "C" void VisualizeSkeleton(GLubyte* ret_bytes, double* targetJoint, double* calibK, uint CameraMode, uint position, float scale, double* floor);
extern "C" void fit_PAF_vec(double* targetJoint2d, double* PAF_vec, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint=nullptr, uint regressor_type=0u, bool quan=false, bool fitPAFfirst=false, bool fit_face_exp=false);
extern "C" void reconstruct_adam(double* pose, double* coeff, double* trans, double* return_joints, int regressor_type=0, bool euler=true);
extern "C" void reconstruct_adam_mesh(double* pose, double* coeff, double* trans, double* face_coeff, int regressor_type=0, bool euler=true);
extern "C" void set_calibK(double* K);
extern "C" void adam_refit(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, int regressor_type);
extern "C" void adam_sequence_init(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, int regressor_type);
extern "C" void adam_hsiu_fit_dome(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, bool bfreezeShape);
extern "C" void refit_eval_h36m(double* pose, double* coeff, double* trans, uint regressor_type=0u, double prior_weight=1.0);
extern "C" void fitSingleStage(double* targetJoint2d, double* PAF_vec, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff, int regressor_type=0u, bool fit_face_exp=false);

std::shared_ptr<Renderer> render = NULL;
std::unique_ptr<VisualizedData> p_vis_data;
std::unique_ptr<ModelFitter> p_modelfitter;
double gResultJoint[21 * 3 + 2 * 21 * 3];
bool model_init = false;
smpl::HandModel g_handl_model;
TotalModel g_total_model;
const int num_body_joint = 20;
const int num_hand_joint = 21;
const int num_face_landmark = 70;

extern "C" void fit_hand3d(double* targetJoint, char* hand_model_file, double* pose, double* coeff, double* trans, int regressor_type, bool euler)
{
	if (!model_init)
	{
		smpl::LoadHandModelFromJson( g_handl_model, string(hand_model_file) );
		model_init = !model_init;
	}

	// Eigen::MatrixXd Joints(3, smpl::HandModel::NUM_JOINTS);
	Eigen::MatrixXd Joints(5, smpl::HandModel::NUM_JOINTS);
	Eigen::MatrixXd surface_constraints(6, 0);
	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
	{
		Joints(0, i) = targetJoint[3 * i];
		Joints(1, i) = targetJoint[3 * i + 1];
		Joints(2, i) = targetJoint[3 * i + 2];
		Joints(3, i) = 0.0;
		Joints(4, i) = 0.0;
	}
	Joints = Joints / SMPL_VIS_SCALING;

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.handl_t.data());
	std::copy(pose, pose + 63, frame_params.handl_pose.data());
	std::copy(coeff, coeff + 63, frame_params.hand_coeffs.data());

	FitToHandCeres_Right_Naive(g_handl_model, Joints, surface_constraints, frame_params.handl_t, frame_params.handl_pose, frame_params.hand_coeffs, regressor_type, false, euler, 1e-2);

	std::copy(frame_params.handl_t.data(), frame_params.handl_t.data() + 3, trans);
	std::copy(frame_params.handl_pose.data(), frame_params.handl_pose.data() + 63, pose);
	std::copy(frame_params.hand_coeffs.data(), frame_params.hand_coeffs.data() + 63, coeff);
}

extern "C" void Opengl_visualize(char* hand_model_file, GLubyte* ret_bytes, double* pose, double* coeff, double* trans, double* targetJoint, bool CameraMode, uint position, int regressor_type, bool stay, bool euler)
{
	if (!model_init)
	{
		smpl::LoadHandModelFromJson( g_handl_model, string(hand_model_file) );
		model_init = !model_init;
	}
	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.handl_t.data());
	std::copy(pose, pose + 63, frame_params.handl_pose.data());
	std::copy(coeff, coeff + 63, frame_params.hand_coeffs.data());

	double resultJoint[63];
	CMeshModelInstance mesh;
	GenerateMesh(mesh, resultJoint, frame_params, g_handl_model, regressor_type, euler);
	// mesh and resultJoint in centimeter

	VisualizedData g_vis_data;
	CopyMesh(mesh, g_vis_data);
	g_vis_data.vis_type = 0;
	g_vis_data.targetJoint = targetJoint;
	g_vis_data.resultJoint = resultJoint;
	g_vis_data.read_buffer = ret_bytes;

	int argc = 0;
	char* argv[1] = {NULL};
	if (render == NULL) render = std::make_shared<Renderer>(&argc, argv);
	Renderer::use_color_fbo = true;
	if (CameraMode)
	{
		assert(render->options.K != NULL);
		render->CameraMode(); // assume options.K is set when we fit 2D hand
	}
	else
	{
		render->NormalMode(position);
		render->options.nRange = 30.0;
	}
	render->options.meshSolid = true;
	render->options.show_joint = false;
	render->RenderHand(g_vis_data);
	if (stay) render->Display();
	render->RenderAndRead();
}

extern "C" void fit_hand2d(double* targetJoint2d, double* calibK, double* PAF_array, char* hand_model_file, double* pose, double* coeff, double* trans, int regressor_type, bool euler, double prior_weight, int mode)
{
	if (!model_init)
	{
		smpl::LoadHandModelFromJson( g_handl_model, string(hand_model_file) );
		model_init = !model_init;
	}

	// Eigen::MatrixXd Joints2d(2, smpl::HandModel::NUM_JOINTS);
	// for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
	// {
	// 	Joints2d(0, i) = targetJoint2d[2 * i];
	// 	Joints2d(1, i) = targetJoint2d[2 * i + 1];
	// }

	Eigen::MatrixXd Joints2d(5, smpl::HandModel::NUM_JOINTS);
	Eigen::MatrixXd surface_constraints(6, 0);
	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; i++)
	{
		Joints2d(0, i) = Joints2d(1, i) = Joints2d(2, i) = 0.0;
		Joints2d(3, i) = targetJoint2d[2 * i];
		Joints2d(4, i) = targetJoint2d[2 * i + 1];
	}
	Eigen::MatrixXd PAF(3, 20);
	std::copy(PAF_array, PAF_array + 20 * 3, PAF.data());

	render->options.K = calibK; // set the K for visualization

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.handl_t.data());
	std::copy(pose, pose + 63, frame_params.handl_pose.data());
	std::copy(coeff, coeff + 63, frame_params.hand_coeffs.data());

	FitToProjectionCeres(g_handl_model, Joints2d, calibK, PAF, surface_constraints, frame_params.handl_t, frame_params.handl_pose, frame_params.hand_coeffs, regressor_type, false, euler, prior_weight, mode);

	std::copy(frame_params.handl_t.data(), frame_params.handl_t.data() + 3, trans);
	std::copy(frame_params.handl_pose.data(), frame_params.handl_pose.data() + 63, pose);
	std::copy(frame_params.hand_coeffs.data(), frame_params.hand_coeffs.data() + 63, coeff);
}

extern "C" void extract_fit_result(char* hand_model_file, double* pose, double* coeff, double* trans, double* resultJoint, int regressor_type, bool euler) 
{
	if (!model_init)
	{
		smpl::LoadHandModelFromJson( g_handl_model, string(hand_model_file) );
		model_init = !model_init;
	}

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.handl_t.data());
	std::copy(pose, pose + 63, frame_params.handl_pose.data());
	std::copy(coeff, coeff + 63, frame_params.hand_coeffs.data());

	// reconstruct_joints(g_handl_model, frame_params.handl_t.data(), frame_params.hand_coeffs.data(), frame_params.handl_pose.data(), resultJoint);
	// resultJoint in meter
	CMeshModelInstance mesh;
	GenerateMesh(mesh, resultJoint, frame_params, g_handl_model, regressor_type, euler);
}

extern "C" void load_totalmodel(char* obj_file, char* model_file, char* pca_file, char* correspondence_file, char* cocoplus_regressor_file)
{
	LoadTotalModelFromObj( g_total_model, string(obj_file) );
	LoadTotalDataFromJson( g_total_model, string(model_file), string(pca_file), string(correspondence_file) );
	LoadCocoplusRegressor( g_total_model, string(cocoplus_regressor_file) );
	p_modelfitter.reset(new ModelFitter(g_total_model));
}

extern "C" void init_renderer()
{
	int argc = 0;
	char* argv[1] = {NULL};
	if (render == NULL) render = std::make_shared<Renderer>(&argc, argv);
}

extern "C" void fit_total3d(double* targetJoint, double* pose, double* coeff, double* trans, double* face_coeff)
{
	Eigen::MatrixXd bodyJoints(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints(5, 3);// (3, 3);		//Heel, Toe
	// Eigen::MatrixXd Joints_face(0, 0);// (3, landmarks_face.size());
	// Eigen::MatrixXd LHandJoints(3, 0);// (3, HandModel::NUM_JOINTS);
	// Eigen::MatrixXd RHandJoints(3, 0);// (3, HandModel::NUM_JOINTS);
	
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = targetJoint[3 * i];
		bodyJoints(1, i) = targetJoint[3 * i + 1];
		bodyJoints(2, i) = targetJoint[3 * i + 2];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints(0, i) = targetJoint[3 * (i + num_body_joint)];
		LHandJoints(1, i) = targetJoint[3 * (i + num_body_joint) + 1];
		LHandJoints(2, i) = targetJoint[3 * (i + num_body_joint) + 2];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints(0, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint)];
		RHandJoints(1, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 1];
		RHandJoints(2, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 2];
	}

	smpl::SMPLParams frame_params;	
	Adam_FitTotalBodyCeres(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face);

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void fit_total2d(double* targetJoint2d, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff)
{
	// #column = 3 but only using first two columns
	Eigen::MatrixXd bodyJoints2d(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints2d_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints2d(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints2d(5, 3);// (3, 3);		//Heel, Toe
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints2d(3, i) = targetJoint2d[2 * i];
		bodyJoints2d(4, i) = targetJoint2d[2 * i + 1];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints2d(3, i) = targetJoint2d[2 * (i + num_body_joint)];
		LHandJoints2d(4, i) = targetJoint2d[2 * (i + num_body_joint) + 1];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints2d(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint)];
		RHandJoints2d(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint) + 1];
	}
	for (int i = 0; i < num_face_landmark; i++)
	{
		Joints2d_face(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint)];
		Joints2d_face(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint) + 1];
	}

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());
	Adam_FitTotalBodyCeres2d(g_total_model, frame_params, bodyJoints2d, RFootJoints2d, LFootJoints2d, RHandJoints2d, LHandJoints2d, Joints2d_face, calibK);

	render->options.K = calibK;

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void fit_total3d2d(double* targetJoint, double* targetJoint2d, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff)
{
	// #column = 3 but only using first two columns
	Eigen::MatrixXd bodyJoints(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints(5, 3);// (3, 3);		//Heel, Toe
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = targetJoint[3 * i];
		bodyJoints(1, i) = targetJoint[3 * i + 1];
		bodyJoints(2, i) = targetJoint[3 * i + 2];
		bodyJoints(3, i) = targetJoint2d[2 * i];
		bodyJoints(4, i) = targetJoint2d[2 * i + 1];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints(0, i) = targetJoint[3 * (i + num_body_joint)];
		LHandJoints(1, i) = targetJoint[3 * (i + num_body_joint) + 1];
		LHandJoints(2, i) = targetJoint[3 * (i + num_body_joint) + 2];
		LHandJoints(3, i) = targetJoint2d[2 * (i + num_body_joint)];
		LHandJoints(4, i) = targetJoint2d[2 * (i + num_body_joint) + 1];
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints(0, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint)];
		RHandJoints(1, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 1];
		RHandJoints(2, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 2];
		RHandJoints(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint)];
		RHandJoints(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint) + 1];
	}
	for (int i = 0; i < num_face_landmark; i++)
	{
		Joints_face(0, i) = 0;
		Joints_face(1, i) = 0;
		Joints_face(2, i) = 0;
		Joints_face(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint)];
		Joints_face(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint) + 1];
	}

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());
	Adam_FitTotalBodyCeres3d2d(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face, calibK);

	render->options.K = calibK;

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void Total_visualize(GLubyte* ret_bytes, double* targetJoint, uint CameraMode, uint position, bool meshSolid, float scale, int vis_type, bool show_joint)
{
	assert(p_vis_data != nullptr);
	p_vis_data->targetJoint = targetJoint;
	p_vis_data->read_buffer = ret_bytes;
	p_vis_data->vis_type = vis_type;

	int argc = 0;
	char* argv[1] = {NULL};
	if (render == NULL) render = std::make_shared<Renderer>(&argc, argv);
	Renderer::use_color_fbo = true;
	if (CameraMode == 1u)
	{
		assert(render->options.K != NULL);
		render->CameraMode(position); // assume options.K is set when we fit 2D hand
		// render->options.view_dist = 200;
		render->options.view_dist = gResultJoint[2 * 3 + 2]; // rotate the scene around (0, 0, view_dist) when xrot and yrot non-zero
	}
	else if (CameraMode == 0u)
	{
		render->NormalMode(position);
		if (vis_type == 1 || vis_type == 2)
		{
			render->options.nRange = 120;
			render->options.view_dist = 300;
		}
		else
		{
			render->options.nRange = 20;
			render->options.view_dist = 100;
		}
	}
	else
	{
		assert(CameraMode == 2u);
		render->OrthoMode(scale, position);
	}
	render->options.meshSolid = meshSolid;
	render->options.show_joint = show_joint;
	render->RenderHand(*p_vis_data);
	// render->Display();
	render->RenderAndRead();

	// cv::Mat frame = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();
	// cv::flip(frame, frame, 0);
	// cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
	// cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	// cv::imshow( "Display window", frame);
	// cv::waitKey(0);
}

extern "C" void VisualizeSkeleton(GLubyte* ret_bytes, double* targetJoint, double* calibK, uint CameraMode, uint position, float scale, double* floor)
{
	VisualizedData g_vis_data;
	g_vis_data.targetJoint = targetJoint;
	g_vis_data.resultJoint = nullptr;
	g_vis_data.read_buffer = ret_bytes;
	g_vis_data.vis_type = 2;

	const Eigen::Map<const Eigen::VectorXd> vFloor(floor, 6);
	const bool bFloor = vFloor.any();
	g_vis_data.bRenderFloor = bFloor;
	if (bFloor)
	{
		g_vis_data.ground_normal[0] = floor[0];
		g_vis_data.ground_normal[1] = floor[1];
		g_vis_data.ground_normal[2] = floor[2];
		g_vis_data.ground_center[0] = floor[3];
		g_vis_data.ground_center[1] = floor[4];
		g_vis_data.ground_center[2] = floor[5];
	}
	else
	{
		std::fill(g_vis_data.ground_center.data(), g_vis_data.ground_center.data() + 3, 0.0);
		std::fill(g_vis_data.ground_normal.data(), g_vis_data.ground_normal.data() + 3, 0.0);
	}

	if (render == NULL)
	{
		int argc = 0;
		char* argv[1] = {NULL};
		render = std::make_shared<Renderer>(&argc, argv);
	}
	Renderer::use_color_fbo = true;
	if (CameraMode == 1u)
	{
		render->options.K = calibK;
		render->CameraMode(); // assume options.K is set when we fit 2D hand
	}
	else if (CameraMode == 0u)
	{
		render->NormalMode(position);
	}
	else
	{
		assert(CameraMode == 2u);
		render->OrthoMode(scale, position);
	}
	render->RenderHandSimple(g_vis_data);
	render->RenderAndRead();
}

extern "C" void fit_PAF_vec(double* targetJoint2d, double* PAF_vec, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff,
							double* targetJoint, uint regressor_type, bool quan, bool fitPAFfirst, bool fit_face_exp)
{
	const uint num_body_joint = 21;
	const uint num_PAF = 63;
	// #column = 3 but only using first two columns
	Eigen::MatrixXd bodyJoints2d(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints2d_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints2d(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints2d(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd PAF(3, num_PAF);
	Eigen::MatrixXd surface_constraint(3, 0);
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints2d(3, i) = targetJoint2d[2 * i];
		bodyJoints2d(4, i) = targetJoint2d[2 * i + 1];
		if (targetJoint)
		{
			bodyJoints2d(0, i) = targetJoint[3 * i + 0];
			bodyJoints2d(1, i) = targetJoint[3 * i + 1];
			bodyJoints2d(2, i) = targetJoint[3 * i + 2];
		}
		else
			bodyJoints2d(0, i) = bodyJoints2d(1, i) = bodyJoints2d(2, i) = 0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints2d(3, i) = targetJoint2d[2 * (i + num_body_joint)];
		LHandJoints2d(4, i) = targetJoint2d[2 * (i + num_body_joint) + 1];
		if (targetJoint)
		{
			LHandJoints2d(0, i) = targetJoint[3 * (i + num_body_joint)];
			LHandJoints2d(1, i) = targetJoint[3 * (i + num_body_joint) + 1];
			LHandJoints2d(2, i) = targetJoint[3 * (i + num_body_joint) + 2];
		}
		else
			LHandJoints2d(0, i) = LHandJoints2d(1, i) = LHandJoints2d(2, i) = 0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints2d(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint)];
		RHandJoints2d(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint) + 1];
		if (targetJoint)
		{
			RHandJoints2d(0, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint)];
			RHandJoints2d(1, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 1];
			RHandJoints2d(2, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 2];
		}
		else
			RHandJoints2d(0, i) = RHandJoints2d(1, i) = RHandJoints2d(2, i) = 0;
	}
	for (int i = 0; i < num_face_landmark; i++)
	{
		Joints2d_face(3, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint)];
		Joints2d_face(4, i) = targetJoint2d[2 * (i + num_hand_joint + num_body_joint + num_hand_joint) + 1];
		Joints2d_face(0, i) = Joints2d_face(1, i) = Joints2d_face(2, i) = 0.0;
	}
	for (int i = 0; i < num_PAF; i++)
	{
		if (PAF_vec[3 * i + 0] != 0.0 || PAF_vec[3 * i + 1] != 0.0 || PAF_vec[3 * i + 2] != 0.0)
		{
			const auto length = sqrt(PAF_vec[3 * i + 0] * PAF_vec[3 * i + 0] + PAF_vec[3 * i + 1] * PAF_vec[3 * i + 1] + PAF_vec[3 * i + 2] * PAF_vec[3 * i + 2]);
			PAF(0, i) = PAF_vec[3 * i + 0] / length;
			PAF(1, i) = PAF_vec[3 * i + 1] / length;
			PAF(2, i) = PAF_vec[3 * i + 2] / length;
		}
		else PAF.col(i).setZero();
	}
	LFootJoints2d.setZero(); RFootJoints2d.setZero();
	LFootJoints2d(3, 0) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0];  // Left BigToe
	LFootJoints2d(4, 0) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1];  // Left BigToe
	LFootJoints2d(3, 1) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2];  // Left SmallToe
	LFootJoints2d(4, 1) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3];  // Left SmallToe
	LFootJoints2d(3, 2) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4];  // Left Heel
	LFootJoints2d(4, 2) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5];  // Left Heel
	RFootJoints2d(3, 0) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0 + 6];  // Right BigToe
	RFootJoints2d(4, 0) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1 + 6];  // Right BigToe
	RFootJoints2d(3, 1) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2 + 6];  // Right SmallToe
	RFootJoints2d(4, 1) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3 + 6];  // Right SmallToe
	RFootJoints2d(3, 2) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4 + 6];  // Right Heel
	RFootJoints2d(4, 2) = targetJoint2d[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5 + 6];  // Right Heel

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	Adam_Fit_PAF(g_total_model, frame_params, bodyJoints2d, RFootJoints2d, LFootJoints2d, RHandJoints2d, LHandJoints2d, Joints2d_face, PAF, surface_constraint,
				 calibK, regressor_type, quan, fitPAFfirst, fit_face_exp);

	render->options.K = calibK;
	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void reconstruct_adam(double* pose, double* coeff, double* trans, double* return_joints, int regressor_type, bool euler)
{
	// Notice that global translation is not used here! It does not matter for H3.6M evalutation.
	const Eigen::Map< const Eigen::Matrix<double, Eigen::Dynamic, 1> > c_bodyshape(coeff, TotalModel::NUM_SHAPE_COEFFICIENTS);
	const Eigen::VectorXd J_vec = g_total_model.J_mu_ + g_total_model.dJdc_ * c_bodyshape;
	Eigen::VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS);
	const double* p2t_parameters[2] = { pose, J_vec.data() };
	double* p2t_residuals = transforms_joint.data();
	smpl::PoseToTransform_AdamFull_withDiff p2t(g_total_model, std::array<std::vector<int>, TotalModel::NUM_JOINTS>(), euler); // the parent indexes can be arbitrary (not used)
	p2t.Evaluate(p2t_parameters, p2t_residuals, nullptr);
	if (regressor_type == 0)
	{
		for (int i = 0; i < TotalModel::NUM_JOINTS; i++)
		{
			return_joints[3 * i + 0] = p2t_residuals[3 * TotalModel::NUM_JOINTS * 4 + 3 * i + 0] + trans[0];
			return_joints[3 * i + 1] = p2t_residuals[3 * TotalModel::NUM_JOINTS * 4 + 3 * i + 1] + trans[1];
			return_joints[3 * i + 2] = p2t_residuals[3 * TotalModel::NUM_JOINTS * 4 + 3 * i + 2] + trans[2];
		}
	}
	else if (regressor_type == 1)
	{
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Vt(TotalModel::NUM_VERTICES, 3);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outVerts(TotalModel::NUM_VERTICES, 3);
		Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, 1> > Vt_vec(Vt.data(), 3 * TotalModel::NUM_VERTICES);
		Vt_vec = g_total_model.m_meanshape + g_total_model.m_shapespace_u * c_bodyshape;
		adam_lbs(g_total_model, Vt_vec.data(), transforms_joint, outVerts.data());
		MatrixXdr J_coco = g_total_model.m_cocoplus_reg * outVerts;
		J_coco.rowwise() += Eigen::Map<Eigen::Matrix<double, 1, 3>>(trans);
		std::copy(J_coco.data(), J_coco.data() + 3 * J_coco.rows(), return_joints);
		// The remaining values are not used. (62 - 19)
	}
	else
	{
		assert(regressor_type == 2);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Vt(TotalModel::NUM_VERTICES, 3);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outVerts(TotalModel::NUM_VERTICES, 3);
		Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, 1> > Vt_vec(Vt.data(), 3 * TotalModel::NUM_VERTICES);
		Vt_vec = g_total_model.m_meanshape + g_total_model.m_shapespace_u * c_bodyshape;
		adam_lbs(g_total_model, Vt_vec.data(), transforms_joint, outVerts.data());
		MatrixXdr J_coco = g_total_model.m_small_coco_reg * outVerts;
		std::copy(J_coco.data(), J_coco.data() + 3 * J_coco.rows(), return_joints);
		std::copy(p2t_residuals + 3 * TotalModel::NUM_JOINTS * 4 + 3 * 22, p2t_residuals + 3 * TotalModel::NUM_JOINTS * 4 + 3 * 62,
				  return_joints + 3 * J_coco.rows()); // copy the fingers
		std::copy(outVerts.data() + 14328 * 3, outVerts.data() + 14328 * 3 + 3, return_joints + 3 * (J_coco.rows() + 40 + 0)); //right bigtoe
		std::copy(outVerts.data() + 14288 * 3, outVerts.data() + 14288 * 3 + 3, return_joints + 3 * (J_coco.rows() + 40 + 1)); //right littletoe
		return_joints[3 * (J_coco.rows() + 40 + 2) + 0] = 0.5 * (outVerts.data()[3 * 14357 + 0] + outVerts.data()[3 * 14361 + 0]); // right heel
		return_joints[3 * (J_coco.rows() + 40 + 2) + 1] = 0.5 * (outVerts.data()[3 * 14357 + 1] + outVerts.data()[3 * 14361 + 1]);
		return_joints[3 * (J_coco.rows() + 40 + 2) + 2] = 0.5 * (outVerts.data()[3 * 14357 + 2] + outVerts.data()[3 * 14361 + 2]);
		std::copy(outVerts.data() + 12239 * 3, outVerts.data() + 12239 * 3 + 3, return_joints + 3 * (J_coco.rows() + 40 + 3)); //left bigtoe
		std::copy(outVerts.data() + 12289 * 3, outVerts.data() + 12289 * 3 + 3, return_joints + 3 * (J_coco.rows() + 40 + 4)); //left smalltoe
		return_joints[3 * (J_coco.rows() + 40 + 5) + 0] = 0.5 * (outVerts.data()[3 * 12368 + 0] + outVerts.data()[3 * 12357 + 0]); // left heel
		return_joints[3 * (J_coco.rows() + 40 + 5) + 1] = 0.5 * (outVerts.data()[3 * 12368 + 1] + outVerts.data()[3 * 12357 + 1]);
		return_joints[3 * (J_coco.rows() + 40 + 5) + 2] = 0.5 * (outVerts.data()[3 * 12368 + 2] + outVerts.data()[3 * 12357 + 2]);
	}
	// for (int i = 26; i < 30; i++)
	// 	std::cout << i << ": " << transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * i)
	// 		<< " " << transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * i + 1) << " " << transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * i + 2) << "\n";
}

extern "C" void reconstruct_adam_mesh(double* pose, double* coeff, double* trans, double* face_coeff, int regressor_type, bool euler)
{
	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + 62 * 3, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + 30, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + 200, frame_params.m_adam_facecoeffs_exp.data());

	CMeshModelInstance mesh;
	GenerateMesh(mesh, gResultJoint, frame_params, g_total_model, regressor_type, euler);

	p_vis_data.reset(new VisualizedData());
	CopyMesh(mesh, *p_vis_data);
	p_vis_data->resultJoint = gResultJoint;
}

extern "C" void fit_h36m_groundtruth(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint)
{
	const uint num_body_joint = 20;
	Eigen::MatrixXd bodyJoints2d(5, num_body_joint);// (3, targetJoint.size());
	for (int i = 0; i < num_body_joint; i++)
	{
		if (targetJoint)
		{
			bodyJoints2d(0, i) = targetJoint[3 * i + 0];
			bodyJoints2d(1, i) = targetJoint[3 * i + 1];
			bodyJoints2d(2, i) = targetJoint[3 * i + 2];
		}
		bodyJoints2d(3, i) = 0.0;
		bodyJoints2d(4, i) = 0.0;
	}
	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	Adam_Fit_H36M(g_total_model, frame_params, bodyJoints2d);

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void set_calibK(double* K)
{
	int argc = 0;
	char* argv[1] = {NULL};
	if (render == NULL) render = std::make_shared<Renderer>(&argc, argv);
	render->options.K = K;
}

extern "C" void adam_refit(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, int regressor_type)
{
	const uint num_body_joint = 20;
	const uint num_PAF = 60;
	// #column = 3 but only using first two columns
	Eigen::MatrixXd bodyJoints(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd PAF(3, num_PAF);
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = targetJoint[3 * i + 0];
		bodyJoints(1, i) = targetJoint[3 * i + 1];
		bodyJoints(2, i) = targetJoint[3 * i + 2];
		bodyJoints(3, i) = 0.0;
		bodyJoints(4, i) = 0.0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints(0, i) = targetJoint[3 * (i + num_body_joint)];
		LHandJoints(1, i) = targetJoint[3 * (i + num_body_joint) + 1];
		LHandJoints(2, i) = targetJoint[3 * (i + num_body_joint) + 2];
		LHandJoints(3, i) = 0.0;
		LHandJoints(4, i) = 0.0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints(0, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint)];
		RHandJoints(1, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 1];
		RHandJoints(2, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 2];
		RHandJoints(3, i) = 0.0;
		RHandJoints(4, i) = 0.0;
	}
	Joints_face.setZero();
	LFootJoints.setZero(); RFootJoints.setZero();
	LFootJoints(0, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0];  // Left BigToe
	LFootJoints(1, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1];  // Left BigToe
	LFootJoints(2, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2];  // Left BigToe
	LFootJoints(0, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3];  // Left SmallToe
	LFootJoints(1, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4];  // Left SmallToe
	LFootJoints(2, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5];  // Left SmallToe
	LFootJoints(0, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 6];  // Left Heel
	LFootJoints(1, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 7];  // Left Heel
	LFootJoints(2, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 8];  // Left Heel
	RFootJoints(0, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0 + 9];  // Right BigToe
	RFootJoints(1, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1 + 9];  // Right BigToe
	RFootJoints(2, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2 + 9];  // Right BigToe
	RFootJoints(0, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3 + 9];  // Right SmallToe
	RFootJoints(1, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4 + 9];  // Right SmallToe
	RFootJoints(2, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5 + 9];  // Right SmallToe
	RFootJoints(0, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 6 + 9];  // Right Heel
	RFootJoints(1, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 7 + 9];  // Right Heel
	RFootJoints(2, 2) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 8 + 9];  // Right Heel
	PAF.setZero();

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	Adam_skeletal_refit(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face, PAF, regressor_type);

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void adam_sequence_init(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, int regressor_type)
{
	const uint num_body_joint = 20;
	const uint num_PAF = 60;
	// #column = 3 but only using first two columns
	Eigen::MatrixXd bodyJoints(5, num_body_joint);// (3, targetJoint.size());
	Eigen::MatrixXd Joints_face(5, num_face_landmark);// (3, landmarks_face.size());
	Eigen::MatrixXd LHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd RHandJoints(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
	Eigen::MatrixXd LFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd RFootJoints(5, 3);// (3, 3);		//Heel, Toe
	Eigen::MatrixXd PAF(3, num_PAF);
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = targetJoint[3 * i + 0];
		bodyJoints(1, i) = targetJoint[3 * i + 1];
		bodyJoints(2, i) = targetJoint[3 * i + 2];
		bodyJoints(3, i) = 0.0;
		bodyJoints(4, i) = 0.0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		LHandJoints(0, i) = targetJoint[3 * (i + num_body_joint)];
		LHandJoints(1, i) = targetJoint[3 * (i + num_body_joint) + 1];
		LHandJoints(2, i) = targetJoint[3 * (i + num_body_joint) + 2];
		LHandJoints(3, i) = 0.0;
		LHandJoints(4, i) = 0.0;
	}
	for (int i = 0; i < num_hand_joint; i++)
	{
		RHandJoints(0, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint)];
		RHandJoints(1, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 1];
		RHandJoints(2, i) = targetJoint[3 * (i + num_hand_joint + num_body_joint) + 2];
		RHandJoints(3, i) = 0.0;
		RHandJoints(4, i) = 0.0;
	}
	Joints_face.setZero();
	LFootJoints.setZero(); RFootJoints.setZero();
	RFootJoints(0, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0];  // Right BigToe
	RFootJoints(1, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1];  // Right BigToe
	RFootJoints(2, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2];  // Right SmallToe
	RFootJoints(0, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3];  // Right BigToe
	RFootJoints(1, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4];  // Right BigToe
	RFootJoints(2, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5];  // Right SmallToe
	LFootJoints(0, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0 + 6];  // Left BigToe
	LFootJoints(1, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1 + 6];  // Left BigToe
	LFootJoints(2, 0) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2 + 6];  // Left SmallToe
	LFootJoints(0, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3 + 6];  // Left BigToe
	LFootJoints(1, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4 + 6];  // Left BigToe
	LFootJoints(2, 1) = targetJoint[3 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5 + 6];  // Left SmallToe
	PAF.setZero();

	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	Adam_skeletal_init(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face, PAF, regressor_type);

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void adam_hsiu_fit_dome(double* pose, double* coeff, double* trans, double* face_coeff, double* targetJoint, bool bfreezeShape)
{
	smpl::SMPLParams frame_params;
	std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	Eigen::MatrixXd bodyJoints(5, num_body_joint);// (3, targetJoint.size());
	for (int i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = targetJoint[3 * i + 0];
		bodyJoints(1, i) = targetJoint[3 * i + 1];
		bodyJoints(2, i) = targetJoint[3 * i + 2];
		bodyJoints(3, i) = 0;
		bodyJoints(4, i) = 0;
	}

	Eigen::MatrixXd RFootJoints(5, 3); std::fill(RFootJoints.data(), RFootJoints.data() + 15, 0.0);
	Eigen::MatrixXd LFootJoints(5, 3); std::fill(LFootJoints.data(), LFootJoints.data() + 15, 0.0);
	Eigen::MatrixXd RHandJoints(5, 21); std::fill(RHandJoints.data(), RHandJoints.data() + 105, 0.0);
	Eigen::MatrixXd LHandJoints(5, 21); std::fill(LHandJoints.data(), LHandJoints.data() + 105, 0.0);
	Eigen::MatrixXd Joints_face(5, 70); std::fill(Joints_face.data(), Joints_face.data() + 350, 0.0);
	Eigen::MatrixXd PAF(3, 54); std::fill(PAF.data(), PAF.data() + 54 * 3, 0.0);
	Adam_skeletal_refit(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face, PAF,
						2, bfreezeShape, false);

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}

extern "C" void refit_eval_h36m(double* pose, double* coeff, double* trans, uint regressor_type, double prior_weight)
{
	const uint num_body_joint = 21;
	// const uint num_PAF = 63;
	std::array<double, 20 * 3> reconstruction;   // 19 angjoo's joints + root
	reconstruct_adam(pose, coeff, trans, reconstruction.data(), regressor_type);
	Eigen::MatrixXd bodyJoints(5, num_body_joint);
	// Eigen::MatrixXd Joints_face(5, num_face_landmark); Joints_face.setZero();
	// Eigen::MatrixXd LHandJoints(5, num_hand_joint); LHandJoints.setZero();
	// Eigen::MatrixXd RHandJoints(5, num_hand_joint); RHandJoints.setZero();
	// Eigen::MatrixXd LFootJoints(5, 3); LFootJoints.setZero();
	// Eigen::MatrixXd RFootJoints(5, 3); RFootJoints.setZero();
	// Eigen::MatrixXd PAF(3, num_PAF); PAF.setZero();

	std::cout << g_total_model.m_cocoplus_reg.rows() << std::endl;
	std::array<int, num_body_joint> cocoplus_to_smc = {{12, 14, 19, 9, 10, 11, 3, 4, 5, 8, 7, 6, 2, 1, 0, 15, 17, 16, 18, 13, 0}};
	for (uint i = 0; i < num_body_joint; i++)
	{
		bodyJoints(0, i) = reconstruction[3 * cocoplus_to_smc[i] + 0];
		bodyJoints(1, i) = reconstruction[3 * cocoplus_to_smc[i] + 1];
		bodyJoints(2, i) = reconstruction[3 * cocoplus_to_smc[i] + 2];
		bodyJoints(3, i) = 0;
		bodyJoints(4, i) = 0;
	}
	bodyJoints.col(2).setZero();
	bodyJoints.col(20).setZero();

	smpl::SMPLParams frame_params;
	frame_params.m_adam_pose.setZero();
	frame_params.m_adam_coeffs.setZero();
	frame_params.m_adam_t.setZero();

	// p_modelfitter->setCalibK(render->options.K);
	p_modelfitter->euler = false;
	p_modelfitter->regressor_type = regressor_type;
	// p_modelfitter->multiStageFitting();

	p_modelfitter->fit_face_exp = false;
	p_modelfitter->fit3D = true;
	p_modelfitter->fit2D = false;
	p_modelfitter->fitPAF = false;
	p_modelfitter->initParameters(frame_params);

	p_modelfitter->bodyJoints[0] = bodyJoints;
	p_modelfitter->rFoot[0].setZero();  // large toe, small toe, 
	p_modelfitter->lFoot[0].setZero();
	p_modelfitter->faceJoints[0].setZero();   // 70 joints
	p_modelfitter->lHandJoints[0].setZero();  // 21 joints
	p_modelfitter->rHandJoints[0].setZero();
	p_modelfitter->PAF[0].setZero();   // 63 PAFs
	p_modelfitter->surface_constraint[0].setZero();

	p_modelfitter->wPosePr = prior_weight;
	p_modelfitter->wCoeffRg = 1e-4;
	p_modelfitter->resetFitData();
	p_modelfitter->resetCostFunction();
	p_modelfitter->pCostFunction[0]->toggle_activate(false, false, false);
	p_modelfitter->pCostFunction[0]->toggle_rigid_body(true);
	p_modelfitter->runFitting();
	p_modelfitter->pCostFunction[0]->toggle_activate(true, false, false);
	p_modelfitter->pCostFunction[0]->toggle_rigid_body(false);
	p_modelfitter->runFitting();
	p_modelfitter->pCostFunction[0]->toggle_activate(true, true, true);
	p_modelfitter->runFitting();

	p_modelfitter->readOutParameters(frame_params);
	// Adam_skeletal_refit(g_total_model, frame_params, bodyJoints, RFootJoints, LFootJoints, RHandJoints, LHandJoints, Joints_face, PAF,
	//                 regressor_type, false, false);  // euler = false
	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
}

extern "C" void fitSingleStage(double* targetJoint2d, double* PAF_vec, double* calibK, double* pose, double* coeff, double* trans, double* face_coeff, int regressor_type, bool fit_face_exp)
{
	std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2> data_vector;
	std::copy(targetJoint2d, targetJoint2d + 2 * ModelFitter::NUM_KEYPOINTS_2D, data_vector.data());
	std::copy(PAF_vec, PAF_vec + 3 * ModelFitter::NUM_PAF_VEC, data_vector.data() + 2 * ModelFitter::NUM_KEYPOINTS_2D);

	smpl::SMPLParams frame_params;
	// std::copy(trans, trans + 3, frame_params.m_adam_t.data());
	// std::copy(pose, pose + TotalModel::NUM_POSE_PARAMETERS, frame_params.m_adam_pose.data());
	// std::copy(coeff, coeff + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_params.m_adam_coeffs.data());
	// std::copy(face_coeff, face_coeff + TotalModel::NUM_EXP_BASIS_COEFFICIENTS, frame_params.m_adam_facecoeffs_exp.data());

	p_modelfitter->regressor_type = regressor_type;
	p_modelfitter->setCalibK(calibK);
	p_modelfitter->euler = false;
	p_modelfitter->fit_face_exp = fit_face_exp;
    p_modelfitter->wPosePr = 200.0;
    p_modelfitter->wCoeffRg = 1.0;
	p_modelfitter->setFitDataNetOutput(data_vector);
	p_modelfitter->multiStageFitting();
	p_modelfitter->readOutParameters(frame_params);

	render->options.K = calibK;
	// CMeshModelInstance mesh;
	// GenerateMesh(mesh, gResultJoint, frame_params, g_total_model, 2, false);
	// VisualizedData vis_data;
	// CopyMesh(mesh, vis_data);
	// vis_data.resultJoint = gResultJoint;
	// render->RenderHand(vis_data);
	// render->CameraMode(0);
	// render->options.nRange = 120;
	// render->options.view_dist = 300;
	// render->Display();

	std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans);
	std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose);
	std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeff);
	std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, face_coeff);
}