#include "handm.h"
#include "totalmodel.h"
#include "ceres/ceres.h"

void SetSolverOptions(ceres::Solver::Options *options);

void FitToHandCeres_Right_Naive(
	smpl::HandModel &handr_model,
	Eigen::MatrixXd &Joints,
	Eigen::MatrixXd &surface_constraint,
	Eigen::Vector3d& parm_handr_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type=0,
	bool fit_surface=false,
	bool euler=true,
	float pose_reg=1e-7,
	float coeff_reg=0.0f);

void FitToProjectionCeres(
	smpl::HandModel &handr_model,
	Eigen::MatrixXd &Joints2d,
	const double* K,
	Eigen::MatrixXd &PAF,
	Eigen::MatrixXd &surface_constraint,
	Eigen::Vector3d& parm_handr_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type=0,
	bool fit_surface=false,
	bool euler=true,
	const double prior_weight=100.0,
	const int mode=0);

void Adam_FitTotalBodyCeres( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints);

void Adam_FitTotalBodyCeres2d( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints2d,
	Eigen::MatrixXd &rFoot2d,	   //2points
	Eigen::MatrixXd &lFoot2d,		//2points
	Eigen::MatrixXd &rHandJoints2d,	   //
	Eigen::MatrixXd &lHandJoints2d,		//
	Eigen::MatrixXd &faceJoints2d,
	double* calibK);

void Adam_FitTotalBodyCeres3d2d( TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints,
	double* calibK);

void Adam_FastFit_Initialize(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints);

void Adam_FastFit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	bool verbose=false);

void Adam_Fit_PAF(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	Eigen::MatrixXd &surface_constraint,
	double* K=nullptr,
	uint regressor_type=0u,
	bool quan=false,
	bool fitPAFfirst=false,
	bool fit_face_exp=false,
	bool euler=true);

void Adam_Fit_H36M(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints);

void Adam_skeletal_refit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type=0u,
	bool bFreezeShape=false,
	bool euler=true);

void Adam_skeletal_init(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type=0u);

void Adam_align_mano(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &surface_constraint);

void Adam_refit_batch(TotalModel &adam,
	std::vector<smpl::SMPLParams*> &frame_param,
	std::vector<Eigen::MatrixXd> &BodyJoints,
	std::vector<Eigen::MatrixXd> &rFoot,
	std::vector<Eigen::MatrixXd> &lFoot,
	std::vector<Eigen::MatrixXd> &rHandJoints,
	std::vector<Eigen::MatrixXd> &lHandJoints,
	std::vector<Eigen::MatrixXd> &faceJoints,
	std::vector<Eigen::MatrixXd> &PAF,
	uint regressor_type=0u,
	bool bFreezeShape=false,
	bool euler=true,
	bool bDCT=true);