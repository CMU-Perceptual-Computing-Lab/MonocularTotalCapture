#include "FitToBody.h"
#include "ceres/normal_prior.h"
#include <iostream>
#include "FitCost.h"
#include "AdamFastCost.h"
#include <chrono>
#include "HandFastCost.h"
#include "DCTCost.h"

void FreezeJoint(ceres::Problem& problem, double* dataPtr, int index)
{
	problem.SetParameterLowerBound(dataPtr, index, -0.00001);
	problem.SetParameterUpperBound(dataPtr, index, 0.00001);
}

void SetSolverOptions(ceres::Solver::Options *options) {
	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options->linear_solver_type));
	CHECK(StringToPreconditionerType("jacobi",
		&options->preconditioner_type));
	options->num_linear_solver_threads = 1;
	options->max_num_iterations = 15;
	options->num_threads = 1;
	options->dynamic_sparsity = true;
	options->use_nonmonotonic_steps = true;
	CHECK(StringToTrustRegionStrategyType("levenberg_marquardt",
		&options->trust_region_strategy_type));
}


void FitToHandCeres_Right_Naive(
	smpl::HandModel &hand_model,
	Eigen::MatrixXd &Joints,
	Eigen::MatrixXd &surface_constraint,
	Eigen::Vector3d& parm_handl_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type,
	bool fit_surface,
	bool euler,
	float pose_reg,
	float coeff_reg)
{
	using namespace Eigen;

	ceres::Problem problem;
	ceres::Solver::Options options;

	// Eigen::MatrixXd Joints_part = Joints.block(0, 0, 3, 21);
	// ceres::CostFunction* fit_cost_analytic_ha =
	// 	new ceres::AutoDiffCostFunction<Hand3DCostPose_LBS, smpl::HandModel::NUM_JOINTS * 3, 3, smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_SHAPE_COEFFICIENTS>
	// 	(new Hand3DCostPose_LBS(hand_model, 0, Joints_part)) ;
	Eigen::MatrixXd PAF(3, 20); PAF.setZero();
	HandFastCost* fit_cost_analytic_ha = new HandFastCost(hand_model, Joints, PAF, surface_constraint, true, false, false, fit_surface, nullptr, regressor_type, euler);
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	if (euler)
	{
		ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
			<CoeffsParameterNorm,
			smpl::HandModel::NUM_JOINTS * 3,
			smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
		ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			pose_reg,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			parm_hand_pose.data());
	}
	else
	{
		Eigen::MatrixXd hand_prior_mu(3 * smpl::HandModel::NUM_JOINTS, 1); hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(0, 0) = hand_model.pose_prior_mu_;
		Eigen::MatrixXd hand_prior_A(60, 3 * smpl::HandModel::NUM_JOINTS); hand_prior_A.setZero();
		hand_prior_A.block<60, 60>(0, 0) = hand_model.pose_prior_A_;
		ceres::CostFunction* hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
		ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			1,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			parm_hand_pose.data());
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	cov.setIdentity();
	Eigen::Matrix<double, Eigen::Dynamic, 1> ones(smpl::HandModel::NUM_JOINTS * 3, 1);
	ones.setOnes();
	ceres::CostFunction *hand_coeff_reg = new ceres::NormalPrior(cov, ones);
	ceres::LossFunction *hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
		coeff_reg,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_coeff_reg,
		hand_coeff_reg_loss,
		parm_hand_coeffs.data());

	// ceres::CostFunction *hand_coeff_reg = new ceres::AutoDiffCostFunction
	// 	<CoeffsParameterNorm,
	// 	smpl::HandModel::NUM_JOINTS * 3,
	// 	smpl::HandModel::NUM_JOINTS * 3>(new CoeffsParameterNorm(smpl::HandModel::NUM_JOINTS * 3));
	// ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
	// 	1e-4,
	// 	ceres::TAKE_OWNERSHIP);
	// problem.AddResidualBlock(hand_coeff_reg,
	// 	hand_coeff_reg_loss,
	// 	parm_hand_coeffs.data());

	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; ++i)
	{
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
	}

	SetSolverOptions(&options);
	// options.function_tolerance = 1e-8;
	// options.max_num_iterations = 50;
	// options.use_nonmonotonic_steps = false;
	// options.num_linear_solver_threads = 1;
	// options.minimizer_progress_to_stdout = true;
	options.update_state_every_iteration = true;
	options.max_num_iterations = 30;
	options.max_solver_time_in_seconds = 8200;
	options.use_nonmonotonic_steps = true;
	options.dynamic_sparsity = true;
	options.min_lm_diagonal = 2e7;
	options.minimizer_progress_to_stdout = true;
	if(fit_surface) std::fill(fit_cost_analytic_ha->weight_vertex.data(), fit_cost_analytic_ha->weight_vertex.data() + fit_cost_analytic_ha->weight_vertex.size(), 1.0);

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	// std::cout << "Before: coeff:\n" << parm_hand_coeffs << std::endl;
	// std::cout << "Before: pose:\n" << parm_hand_pose << std::endl;
	// std::cout << "Before: trans:\n" << parm_handl_t << std::endl;
	ceres::Solver::Summary summary;

	problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	// std::cout << "After: coeff:\n" << parm_hand_coeffs << std::endl;
	// std::cout << "After: pose:\n" << parm_hand_pose << std::endl;
	// std::cout << "After: trans:\n" << parm_handl_t << std::endl;

	printf("FitToHandCeres_Right_Naive: Done\n");
}

void FitToProjectionCeres(
	smpl::HandModel &hand_model,
	Eigen::MatrixXd &Joints2d,
	const double* K,
	Eigen::MatrixXd &PAF,
	Eigen::MatrixXd &surface_constraint,
	Eigen::Vector3d& parm_handl_t,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_pose,
	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& parm_hand_coeffs,
	int regressor_type,
	bool fit_surface,
	bool euler,
	const double prior_weight,
	const int mode
)
{
	using namespace Eigen;
	ceres::Problem problem_init;
	ceres::Solver::Options options_init;
	// define the reprojection error
	HandFastCost* fit_cost_analytic_ha_init = new HandFastCost(hand_model, Joints2d, PAF, surface_constraint, false, false, true, false, nullptr, regressor_type, euler);
	problem_init.AddResidualBlock(fit_cost_analytic_ha_init,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	assert(mode >= 0 && mode <= 3);

	// Regularization
	if (euler)
	{
		ceres::CostFunction *hand_pose_reg_init = new ceres::AutoDiffCostFunction
			<HandPoseParameterNorm,
			smpl::HandModel::NUM_JOINTS * 3,
			smpl::HandModel::NUM_JOINTS * 3>(new HandPoseParameterNorm(smpl::HandModel::NUM_JOINTS * 3, hand_model));
		ceres::LossFunction* hand_pose_reg_loss_init = new ceres::ScaledLoss(NULL,
			1e-5,
			ceres::TAKE_OWNERSHIP);
		problem_init.AddResidualBlock(hand_pose_reg_init,
			hand_pose_reg_loss_init,
			parm_hand_pose.data());
	}
	else
	{
		Eigen::MatrixXd hand_prior_mu(3 * smpl::HandModel::NUM_JOINTS, 1); hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(0, 0) = hand_model.pose_prior_mu_;
		Eigen::MatrixXd hand_prior_A(60, 3 * smpl::HandModel::NUM_JOINTS); hand_prior_A.setZero();
		hand_prior_A.block<60, 60>(0, 0) = hand_model.pose_prior_A_;
		ceres::CostFunction* hand_pose_reg_init = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
		ceres::LossFunction* hand_pose_reg_loss_init = new ceres::ScaledLoss(NULL,
			prior_weight,
			ceres::TAKE_OWNERSHIP);
		problem_init.AddResidualBlock(hand_pose_reg_init,
			hand_pose_reg_loss_init,
			parm_hand_pose.data());
	}

	Eigen::MatrixXd A(smpl::HandModel::NUM_JOINTS * 3, smpl::HandModel::NUM_JOINTS * 3);
	A.setIdentity();
	Eigen::VectorXd b(smpl::HandModel::NUM_JOINTS * 3);
	b.setOnes();
	// ceres::CostFunction *hand_coeff_reg_init = new ceres::NormalPrior(A, b);
	// ceres::LossFunction* hand_coeff_reg_loss_init = new ceres::ScaledLoss(NULL,
	// 	1000,
	// 	ceres::TAKE_OWNERSHIP);
	// problem_init.AddResidualBlock(hand_coeff_reg_init,
	// 	hand_coeff_reg_loss_init,
	// 	parm_hand_coeffs.data());

	for (int i = 0; i < smpl::HandModel::NUM_JOINTS - 1; ++i)
	{
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem_init.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 0, 2);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 1, 2);
		problem_init.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 2, 2);
	}

	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-8;
	options_init.max_num_iterations = 30;
	options_init.use_nonmonotonic_steps = true;
	options_init.num_linear_solver_threads = 1;
	options_init.minimizer_progress_to_stdout = true;
	options_init.parameter_tolerance = 1e-12;

	if (mode == 0)
		problem_init.SetParameterBlockConstant(parm_hand_coeffs.data());
	else if (mode == 1)
	{
		problem_init.SetParameterBlockConstant(parm_hand_coeffs.data());
		problem_init.SetParameterBlockConstant(parm_hand_pose.data());
	}
	else if (mode == 2)
	{
		problem_init.SetParameterBlockConstant(parm_hand_pose.data());
	}
	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options_init.linear_solver_type));
	ceres::Solver::Summary summary_init;
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << "\n";
	// problem_init.SetParameterBlockVariable(parm_hand_coeffs.data());

	ceres::Problem problem;
	ceres::Solver::Options options;
	HandFastCost* fit_cost_analytic_ha = new HandFastCost(hand_model, Joints2d, PAF, surface_constraint, false, true, true, fit_surface, K, regressor_type, euler);
	fit_cost_analytic_ha->weight_PAF = 50.0f;
	problem.AddResidualBlock(fit_cost_analytic_ha,
		NULL,
		parm_handl_t.data(),
		parm_hand_pose.data(),
		parm_hand_coeffs.data());

	// Regularization
	if (euler)
	{
		ceres::CostFunction *hand_pose_reg = new ceres::AutoDiffCostFunction
			<HandPoseParameterNorm,
			smpl::HandModel::NUM_JOINTS * 3,
			smpl::HandModel::NUM_JOINTS * 3>(new HandPoseParameterNorm(smpl::HandModel::NUM_JOINTS * 3, hand_model));
		ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			1e-5,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			parm_hand_pose.data());
	}
	else
	{
		Eigen::MatrixXd hand_prior_mu(3 * smpl::HandModel::NUM_JOINTS, 1); hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(0, 0) = hand_model.pose_prior_mu_;
		Eigen::MatrixXd hand_prior_A(60, 3 * smpl::HandModel::NUM_JOINTS); hand_prior_A.setZero();
		hand_prior_A.block<60, 60>(0, 0) = hand_model.pose_prior_A_;
		ceres::CostFunction* hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
		ceres::LossFunction* hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			prior_weight,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			parm_hand_pose.data());
	}

	ceres::CostFunction *hand_coeff_reg = new ceres::NormalPrior(A, b);
	ceres::LossFunction* hand_coeff_reg_loss = new ceres::ScaledLoss(NULL,
		10,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(hand_coeff_reg,
		hand_coeff_reg_loss,
		parm_hand_coeffs.data());

	for (int i = 0; i < smpl::HandModel::NUM_JOINTS; ++i)
	{
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 0, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 1, 0.5);
		problem.SetParameterLowerBound(parm_hand_coeffs.data(), i * 3 + 2, 0.5);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 0, 2);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 1, 2);
		problem.SetParameterUpperBound(parm_hand_coeffs.data(), i * 3 + 2, 2);
	}

	SetSolverOptions(&options);
	options.function_tolerance = 1e-8;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;
	options.parameter_tolerance = 1e-12;

	CHECK(StringToLinearSolverType("sparse_normal_cholesky",
		&options.linear_solver_type));
	ceres::Solver::Summary summary;
	if (mode == 0)
		problem.SetParameterBlockConstant(parm_hand_coeffs.data());
	else if (mode == 1)
	{
		problem.SetParameterBlockConstant(parm_hand_coeffs.data());
		problem.SetParameterBlockConstant(parm_hand_pose.data());
		// fit_cost_analytic_ha->weight_2d[0] = 0.0;
	}
	else if (mode == 2)
	{
		problem.SetParameterBlockConstant(parm_hand_pose.data());
	}
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	// std::cout << "Fit shape: coeff" << parm_hand_coeffs << std::endl;
	// std::cout << "Fit shape: pose" << parm_hand_pose << std::endl;
	// std::cout << "Fit shape: trans" << parm_handl_t << std::endl;
	// problem.SetParameterBlockVariable(parm_hand_coeffs.data());
	// ceres::Solve(options, &problem, &summary);
	std::cout << "After: coeff" << parm_hand_coeffs << std::endl;
	std::cout << "After: pose" << parm_hand_pose << std::endl;
	std::cout << "After: trans" << parm_handl_t << std::endl;

	printf("FitToProjectionCeres: Done\n");
}


void Adam_FitTotalBodyCeres(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints)
{
	using namespace Eigen;
	MatrixXd PAF(3, 54);
	MatrixXd surface_constraint(6, 0);
	// std::fill(PAF.data(), PAF.data() + PAF.size(), 0);
	const AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true);
	ceres::Problem problem_init;
	AdamFullCost* adam_cost = new AdamFullCost(data);

	problem_init.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs_init = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose_init = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options_init);
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 1;
	options_init.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_rigid_body(false);
	adam_cost->toggle_activate(true, false, false);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_activate(true, true, true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;
}

void Adam_FitTotalBodyCeres2d(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints2d,
	Eigen::MatrixXd &rFoot2d,	   //2points
	Eigen::MatrixXd &lFoot2d,		//2points
	Eigen::MatrixXd &rHandJoints2d,	   //
	Eigen::MatrixXd &lHandJoints2d,		//
	Eigen::MatrixXd &faceJoints2d,
	double* calibK)
{
	using namespace Eigen;

	ceres::Problem problem_init;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints_init = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints2d, rFoot2d, lFoot2d, faceJoints2d,
																								 lHandJoints2d, rHandJoints2d, calibK, false, 1u);

	problem_init.AddResidualBlock(cost_body_keypoints_init,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary_init;
	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-4;
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 1;
	options_init.minimizer_progress_to_stdout = true;
	cost_body_keypoints_init->joint_only = true;
	// Pure Translation, fit body only!
	cost_body_keypoints_init->toggle_activate(false, false);
	problem_init.SetParameterBlockConstant(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());

	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	// Tranlation, pose and shape, add regularization, use body and hand
	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs_init = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_init = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	cost_body_keypoints_init->toggle_activate(true, true);
	problem_init.SetParameterBlockVariable(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockVariable(frame_param.m_adam_coeffs.data());
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints2d, rFoot2d, lFoot2d, faceJoints2d,
																								 lHandJoints2d, rHandJoints2d, calibK, true, 1u);

	problem.AddResidualBlock(cost_body_keypoints,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data(),
		frame_param.m_adam_facecoeffs_exp.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	// ceres::CostFunction *cost_prior_face_exp = new ceres::NormalPrior(facem.face_prior_A_exp, facem.face_prior_mu_exp);
	ceres::CostFunction *cost_prior_face_exp = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_EXP_BASIS_COEFFICIENTS));
	ceres::LossFunction *loss_weight_prior_face_exp = new ceres::ScaledLoss(NULL,
		10000,		//original
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_face_exp,
		loss_weight_prior_face_exp,
		frame_param.m_adam_facecoeffs_exp.data());
	
	// problem.SetParameterBlockConstant(frame_param.m_adam_facecoeffs_exp.data());
	options.max_num_iterations = 10;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;
	cost_body_keypoints->joint_only = false;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}

void Adam_FitTotalBodyCeres3d2d(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints,
	double* calibK)
{
	using namespace Eigen;

	int weight2d = 1.0f;

	ceres::Problem problem_init;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints_init = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints,
																								 lHandJoints, rHandJoints, calibK, false, 2u);

	problem_init.AddResidualBlock(cost_body_keypoints_init,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary_init;
	SetSolverOptions(&options_init);
	options_init.function_tolerance = 1e-4;
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 1;
	options_init.minimizer_progress_to_stdout = true;
	cost_body_keypoints_init->joint_only = true;
	// Pure Translation, fit body only!
	cost_body_keypoints_init->toggle_activate(true, false);
	cost_body_keypoints_init->weight2d = weight2d;
	problem_init.SetParameterBlockConstant(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());

	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	// Tranlation, pose and shape, add regularization, use body and hand
	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs_init = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose_init = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	cost_body_keypoints_init->toggle_activate(true, true);
	problem_init.SetParameterBlockVariable(frame_param.m_adam_pose.data());
	problem_init.SetParameterBlockVariable(frame_param.m_adam_coeffs.data());
	ceres::Solve(options_init, &problem_init, &summary_init);
	std::cout << summary_init.FullReport() << std::endl;

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	CostFunc_Adam_keypoints_withFoot *cost_body_keypoints = new CostFunc_Adam_keypoints_withFoot(adam, BodyJoints, rFoot, lFoot, faceJoints,
																								 lHandJoints, rHandJoints, calibK, true, 2u);

	problem.AddResidualBlock(cost_body_keypoints,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data(),
		frame_param.m_adam_facecoeffs_exp.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_coeffs = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_SHAPE_COEFFICIENTS,
		TotalModel::NUM_SHAPE_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_SHAPE_COEFFICIENTS));
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		<AdamBodyPoseParamPrior,
		TotalModel::NUM_POSE_PARAMETERS,
		TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));

	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	// ceres::CostFunction *cost_prior_face_exp = new ceres::NormalPrior(facem.face_prior_A_exp, facem.face_prior_mu_exp);
	ceres::CostFunction *cost_prior_face_exp = new ceres::AutoDiffCostFunction
		<CoeffsParameterNorm,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS,
		TotalModel::NUM_EXP_BASIS_COEFFICIENTS>(new CoeffsParameterNorm(TotalModel::NUM_EXP_BASIS_COEFFICIENTS));
	ceres::LossFunction *loss_weight_prior_face_exp = new ceres::ScaledLoss(NULL,
		10000,		//original
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_face_exp,
		loss_weight_prior_face_exp,
		frame_param.m_adam_facecoeffs_exp.data());
	
	// problem.SetParameterBlockConstant(frame_param.m_adam_facecoeffs_exp.data());
	options.max_num_iterations = 10;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;
	cost_body_keypoints->joint_only = false;
	cost_body_keypoints->weight2d = weight2d;
	// ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << std::endl;
}

void Adam_FastFit_Initialize(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints)
{
	using namespace Eigen;
	MatrixXd PAF(3, 54);
	MatrixXd surface_constraint(6, 0);
	// std::fill(PAF.data(), PAF.data() + PAF.size(), 0);
	const AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true);
	ceres::Problem problem_init;
	AdamFullCost* adam_cost = new AdamFullCost(data);

	problem_init.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());	

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs_init = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs_init = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_coeffs_init,
		loss_weight_prior_body_coeffs_init,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose_init = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose_init = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	problem_init.AddResidualBlock(cost_prior_body_pose_init,
		loss_weight_prior_body_pose_init,
		frame_param.m_adam_pose.data());

	for (auto d = 0; d < 5; d++)
	{
		adam_cost->m_targetPts_weight[5 * (adam_cost->m_nCorrespond_adam2joints + 0) + d] = 
		adam_cost->m_targetPts_weight[5 * (adam_cost->m_nCorrespond_adam2joints + 1) + d] = 
		adam_cost->m_targetPts_weight[5 * (adam_cost->m_nCorrespond_adam2joints + 2) + d] = 1.0;
		adam_cost->m_targetPts_weight[5 * (adam_cost->m_nCorrespond_adam2joints + 2) + d] = 0;
	}
	ceres::Solver::Options options_init;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options_init);
	options_init.max_num_iterations = 20;
	options_init.use_nonmonotonic_steps = false;
	options_init.num_linear_solver_threads = 1;
	options_init.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_rigid_body(false);
	adam_cost->toggle_activate(true, false, false);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_activate(true, true, true);
	ceres::Solve(options_init, &problem_init, &summary);
	std::cout << summary.FullReport() << std::endl;
}

ceres::Problem g_problem;
smpl::SMPLParams g_params;
AdamFastCost* g_cost_body_keypoints = NULL;
void Adam_FastFit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,	   //2points
	Eigen::MatrixXd &lFoot,		//2points
	Eigen::MatrixXd &rHandJoints,	   //
	Eigen::MatrixXd &lHandJoints,		//
	Eigen::MatrixXd &faceJoints,
	bool verbose)
{
// const auto start1 = std::chrono::high_resolution_clock::now();
	// use the existing shape coeff in frame_param, fit the pose and trans fast
	using namespace Eigen;
	if (g_cost_body_keypoints == NULL)
	{
		g_cost_body_keypoints = new AdamFastCost(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, frame_param.m_adam_coeffs.data());
		g_problem.AddResidualBlock(g_cost_body_keypoints,
			NULL,
			g_params.m_adam_t.data(),
			g_params.m_adam_pose.data()
		);	

		//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
		// ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
		// 	<AdamBodyPoseParamPrior,
		// 	TotalModel::NUM_POSE_PARAMETERS,
		// 	TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
		AdamBodyPoseParamPriorDiff *cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
		for (int i = 22; i < 62; i++)
		{
			cost_prior_body_pose->weight[3 * i + 2] = 1e-2; // set regularization of finger bend small
		}

		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			1e-2,
			ceres::TAKE_OWNERSHIP);
		g_problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			g_params.m_adam_pose.data());

		std::copy(frame_param.m_adam_coeffs.data(), frame_param.m_adam_coeffs.data() + 30, g_params.m_adam_coeffs.data());  // always use the shape coeff of first frame
	}
	else g_cost_body_keypoints->UpdateJoints(BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints);
// const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start1).count();
// const auto start2 = std::chrono::high_resolution_clock::now();

	auto twist = false;
	for (int i = 1; i < TotalModel::NUM_JOINTS; i++)
	{
		auto count_axis = 0u;
		if (frame_param.m_adam_pose.data()[3 * i + 0] > 90 || frame_param.m_adam_pose.data()[3 * i + 0] < -90) count_axis++;
		if (frame_param.m_adam_pose.data()[3 * i + 1] > 90 || frame_param.m_adam_pose.data()[3 * i + 1] < -90) count_axis++;
		if (frame_param.m_adam_pose.data()[3 * i + 2] > 90 || frame_param.m_adam_pose.data()[3 * i + 2] < -90) count_axis++;
		if (count_axis >= 2)  // this joint is twisted
		{
			twist = true;
			frame_param.m_adam_pose.data()[3 * i + 0] = frame_param.m_adam_pose.data()[3 * i + 1] = frame_param.m_adam_pose.data()[3 * i + 2] = 0.0;
		}
	}

	std::copy(frame_param.m_adam_t.data(), frame_param.m_adam_t.data() + 3, g_params.m_adam_t.data());
	std::copy(frame_param.m_adam_pose.data(), frame_param.m_adam_pose.data() + 62 * 3, g_params.m_adam_pose.data());
// const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start2).count();
// const auto start3 = std::chrono::high_resolution_clock::now();

	ceres::Solver::Options options;
	SetSolverOptions(&options);
	options.max_num_iterations = 20;
	options.use_nonmonotonic_steps = true;
	options.minimizer_progress_to_stdout = verbose;
	ceres::Solver::Summary summary;

	// g_problem.SetParameterBlockConstant(g_params.m_adam_t.data());
	// g_problem.SetParameterBlockConstant(g_params.m_adam_pose.data());

	// g_cost_body_keypoints->toggle_activate(false, false);
	// ceres::Solve(options, &g_problem, &summary);
	// std::cout << summary.FullReport() << std::endl;

	if (!twist) g_cost_body_keypoints->toggle_activate(true, true);   // if no twist is detected, perform a single fitting
	else
	{
		std::cout << "twist detected, multiple stage fitting" << std::endl;
		g_cost_body_keypoints->toggle_activate(true, false);
		ceres::Solve(options, &g_problem, &summary);
		if(verbose) std::cout << summary.FullReport() << std::endl;
	}
	ceres::Solve(options, &g_problem, &summary);
	if(verbose) std::cout << summary.FullReport() << std::endl;

	std::copy(g_params.m_adam_t.data(), g_params.m_adam_t.data() + 3, frame_param.m_adam_t.data());
	std::copy(g_params.m_adam_pose.data(), g_params.m_adam_pose.data() + 62 * 3, frame_param.m_adam_pose.data());
// const auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start5).count();
// std::cout << __FILE__ << " " << duration1 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration2 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration3 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration4 * 1e-6 << "\n"
// 		  << __FILE__ << " " << duration5 * 1e-6 << "\n" << std::endl;
}

void Adam_Fit_PAF(TotalModel &adam, smpl::SMPLParams &frame_param, Eigen::MatrixXd &BodyJoints, Eigen::MatrixXd &rFoot, Eigen::MatrixXd &lFoot, Eigen::MatrixXd &rHandJoints,
				  Eigen::MatrixXd &lHandJoints, Eigen::MatrixXd &faceJoints, Eigen::MatrixXd &PAF, Eigen::MatrixXd &surface_constraint, double* calibK,
				  uint regressor_type, bool quan, bool fitPAFfirst, bool fit_face_exp, bool euler)
{
	using namespace Eigen;	
const auto start = std::chrono::high_resolution_clock::now();

	if (fitPAFfirst)  // if true, fit onto only PAF first
	{
		std::cout << "Fitting to 3D skeleton as the first step" << std::endl;
		ceres::Problem init_problem;

		AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint,
	 					 false, false, nullptr, true, false);
		AdamFullCost* adam_cost;
		adam_cost = new AdamFullCost(data, regressor_type, false, euler);

		init_problem.AddResidualBlock(adam_cost,
			NULL,
			frame_param.m_adam_t.data(),
			frame_param.m_adam_pose.data(),
			frame_param.m_adam_coeffs.data());

		// for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++)
		// {
		// 	init_problem.SetParameterLowerBound(frame_param.m_adam_pose.data(), i, -180);
		// 	init_problem.SetParameterUpperBound(frame_param.m_adam_pose.data(), i, 180);
		// }

		ceres::Solver::Options init_options;
		ceres::Solver::Summary init_summary;
		SetSolverOptions(&init_options);
		init_options.function_tolerance = 1e-4;
		init_options.max_num_iterations = 20;
		init_options.use_nonmonotonic_steps = true;
		init_options.num_linear_solver_threads = 1;
		init_options.minimizer_progress_to_stdout = true;
		// if (quan) init_problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
		adam_cost->toggle_activate(false, false, false);
		adam_cost->toggle_rigid_body(true);

const auto start_solve = std::chrono::high_resolution_clock::now();
		ceres::Solve(init_options, &init_problem, &init_summary);
		std::cout << init_summary.FullReport() << std::endl;

		//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
		CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
		ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
			quan? 1e-2 : 1e-2,
			ceres::TAKE_OWNERSHIP);
		init_problem.AddResidualBlock(cost_prior_body_coeffs,
			loss_weight_prior_body_coeffs,
			frame_param.m_adam_coeffs.data());

		//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
		AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			quan? 1e-2 : 1e-1,
			// 1e-2,
			ceres::TAKE_OWNERSHIP);
		std::fill(cost_prior_body_pose->weight.data() + 3, cost_prior_body_pose->weight.data() + 9, 10.);
		std::fill(cost_prior_body_pose->weight.data() + 12, cost_prior_body_pose->weight.data() + 18, 10.);
		init_problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			frame_param.m_adam_pose.data());
		if(!euler)
		{
			Eigen::Matrix<double, 72, TotalModel::NUM_POSE_PARAMETERS> prior_A; prior_A.setZero();
			prior_A.block<72, 66>(0, 0) = adam.smpl_pose_prior_A.block<72, 66>(0, 0);  // for body, use the prior from SMPL
			Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> prior_mu; prior_mu.setZero();
			prior_mu.block<66, 1>(0, 0) = -adam.smpl_pose_prior_mu.block<66, 1>(0, 0);  // use the prior from SMPL (negative)
			ceres::CostFunction *pose_reg = new ceres::NormalPrior(prior_A, prior_mu);
			ceres::LossFunction *pose_reg_loss = new ceres::ScaledLoss(NULL,
				10,
				ceres::TAKE_OWNERSHIP);
			init_problem.AddResidualBlock(pose_reg,
				pose_reg_loss,
				frame_param.m_adam_pose.data());
			for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++) cost_prior_body_pose->weight[i] = 0.0;  // only use regularization for fingers
			Eigen::MatrixXd hand_prior_A(120, TotalModel::NUM_POSE_PARAMETERS); hand_prior_A.setZero();
			Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> hand_prior_mu; hand_prior_mu.setZero();
			hand_prior_mu.block<60, 1>(66, 0) = -adam.hand_pose_prior_mu; hand_prior_mu.block<60, 1>(126, 0) = adam.hand_pose_prior_mu;
			for (int i = 66; i < 126; i += 3) hand_prior_mu(i, 0) = -hand_prior_mu(i, 0);
			hand_prior_A.block<60, 60>(0, 66) = -adam.hand_pose_prior_A; hand_prior_A.block<60, 60>(60, 126) = adam.hand_pose_prior_A;
			for (int i = 66; i < 126; i += 3) hand_prior_A.col(i) = -hand_prior_A.col(i);
			ceres::CostFunction *hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
			ceres::LossFunction *hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
				10,
				ceres::TAKE_OWNERSHIP);
			init_problem.AddResidualBlock(hand_pose_reg,
				hand_pose_reg_loss,
				frame_param.m_adam_pose.data());
		}

		init_options.function_tolerance = 1e-4;
		adam_cost->toggle_activate(true, true, true);
		adam_cost->toggle_rigid_body(false);
		ceres::Solve(init_options, &init_problem, &init_summary);
		std::cout << init_summary.FullReport() << std::endl;

const auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_solve).count();
std::cout << "3D solve time: " << duration_solve * 1e-6 << "\n";
		frame_param.m_adam_t[2] = 200.0; // for fitting onto projection
	}

	std::cout << "Fitting to 2D skeleton Projection" << std::endl;
	ceres::Problem problem;
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, false, true, calibK, true, surface_constraint.cols() > 0);
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type, fit_face_exp, euler);

	if (fit_face_exp)
		problem.AddResidualBlock(adam_cost,
			NULL,
			frame_param.m_adam_t.data(),
			frame_param.m_adam_pose.data(),
			frame_param.m_adam_coeffs.data(),
			frame_param.m_adam_facecoeffs_exp.data());
	else
		problem.AddResidualBlock(adam_cost,
			NULL,
			frame_param.m_adam_t.data(),
			frame_param.m_adam_pose.data(),
			frame_param.m_adam_coeffs.data());

	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	// if(!fitPAFfirst && !quan)  // if fitPAFfirst, should be the first frame in video, allow shape change
	// {
	// 	problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
	// }

const auto start_solve = std::chrono::high_resolution_clock::now();
	if(!quan) // for quantitative, don't solve this time, especially for failure cases
	{
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << std::endl;
	}

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		quan? 1e-5 : 1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		quan? 1e-2 : 1e-2,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());
	if (!euler)
	{
		Eigen::Matrix<double, 72, TotalModel::NUM_POSE_PARAMETERS> prior_A; prior_A.setZero();
		prior_A.block<72, 66>(0, 0) = adam.smpl_pose_prior_A.block<72, 66>(0, 0);  // for body, use the prior from SMPL
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> prior_mu; prior_mu.setZero();
		prior_mu.block<66, 1>(0, 0) = -adam.smpl_pose_prior_mu.block<66, 1>(0, 0);  // use the prior from SMPL
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(prior_A, prior_mu);
		ceres::LossFunction* pose_reg_loss = new ceres::ScaledLoss(NULL,
			10,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(pose_reg,
			pose_reg_loss,
			frame_param.m_adam_pose.data());
		for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++) cost_prior_body_pose->weight[i] = 0.0;  // only use regularization for fingers
		Eigen::MatrixXd hand_prior_A(120, TotalModel::NUM_POSE_PARAMETERS); hand_prior_A.setZero();
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> hand_prior_mu; hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(66, 0) = -adam.hand_pose_prior_mu; hand_prior_mu.block<60, 1>(126, 0) = adam.hand_pose_prior_mu;
		for (int i = 66; i < 126; i += 3) hand_prior_mu(i, 0) = -hand_prior_mu(i, 0);
		hand_prior_A.block<60, 60>(0, 66) = -adam.hand_pose_prior_A; hand_prior_A.block<60, 60>(60, 126) = adam.hand_pose_prior_A;
		for (int i = 66; i < 126; i += 3) hand_prior_A.col(i) = -hand_prior_A.col(i);
		ceres::CostFunction *hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
		ceres::LossFunction *hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			10,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			frame_param.m_adam_pose.data());
	}

	if (fit_face_exp)
	{
		ceres::NormalPrior *cost_prior_face_exp = new ceres::NormalPrior(adam.face_prior_A_exp.asDiagonal(), Eigen::Matrix<double, TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 1>::Zero());
		ceres::LossFunction *loss_weight_prior_face_exp = new ceres::ScaledLoss(NULL,
			100,		//original
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(cost_prior_face_exp,
			loss_weight_prior_face_exp,
			frame_param.m_adam_facecoeffs_exp.data());
	}

	if (quan)
	{
		for (int i = 0; i < 12; i++) adam_cost->PAF_weight[i] = 50;
		if(euler) std::fill(cost_prior_body_pose->weight.data() + 36, cost_prior_body_pose->weight.data() + TotalModel::NUM_POSE_PARAMETERS, 2.0);
	}
	else
	{
		if (regressor_type == 0)
		{
			// setting for make videos
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 6] =
			// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 7] = 1.0;
			std::fill(adam_cost->m_targetPts_weight.data() + 5 * adam_cost->m_nCorrespond_adam2joints,
    				  adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + 8), 1.0);
			// for (auto i = 0; i < adam.m_correspond_adam2face70_adamIdx.rows(); i++)  // face starts from 8
			// 	adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 8] = 1.0;
			std::fill(adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + 8),
					  adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + adam.m_correspond_adam2face70_adamIdx.rows()), 1.0);
		}
		else if (regressor_type == 2)
		{
			// set weight for all vertices
			// for (auto i = 0; i < adam_cost->m_nCorrespond_adam2pts; ++i)
			// {
			// 	if (i < 8 + adam.m_correspond_adam2face70_adamIdx.rows()) adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + i] = 1.0;
			// 	else adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + i] = 0.1; // remaining are surface constraints
			// }
			std::fill(adam_cost->m_targetPts_weight.data() + 5 * adam_cost->m_nCorrespond_adam2joints,
    				  adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + 8 + adam.m_correspond_adam2face70_adamIdx.rows()), 1.0);
			std::fill(adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + 8 + adam.m_correspond_adam2face70_adamIdx.rows()),
    				  adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + adam_cost->m_nCorrespond_adam2pts), 0.1);
		}
		adam_cost->toggle_activate(true, false, false);
		adam_cost->toggle_rigid_body(false);
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << std::endl;
	}

	adam_cost->toggle_activate(true, true, true);
	adam_cost->toggle_rigid_body(false);
	if (!quan && regressor_type == 2)
	{
		adam_cost->PAF_weight[12] = 5;
		adam_cost->PAF_weight[13] = 5;
		// adam_cost->m_targetPts_weight[17] = 0.1;
		// adam_cost->m_targetPts_weight[18] = 0.1;
		std::fill(adam_cost->m_targetPts_weight.data() + 17 * 5, adam_cost->m_targetPts_weight.data() + 19 * 5, 0.1);  // lower cost for ear, satisfy the face70 constraints
	}
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
const auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_solve).count();
std::cout << "2D solve time: " << duration_solve * 1e-6 << "\n";

const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
std::cout << "Total fitting time: " << duration * 1e-6 << "\n";
}

void Adam_Fit_H36M(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints)
{
	ceres::Problem init_problem;
	Eigen::MatrixXd faceJoints(5, 70);
	Eigen::MatrixXd lHandJoints(5, 20);
	Eigen::MatrixXd rHandJoints(5, 20);
	Eigen::MatrixXd lFoot(5, 3);
	Eigen::MatrixXd rFoot(5, 3);
	Eigen::MatrixXd PAF(3, 14);
	Eigen::MatrixXd surface_constraint(6, 0);
	faceJoints.setZero();
	lHandJoints.setZero();
	rHandJoints.setZero();
	lFoot.setZero();
	rFoot.setZero();
	PAF.setZero();
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true, false, nullptr, false, false);
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, 1);

	init_problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	ceres::Solver::Options init_options;
	ceres::Solver::Summary init_summary;
	SetSolverOptions(&init_options);
	init_options.function_tolerance = 1e-4;
	init_options.max_num_iterations = 30;
	init_options.use_nonmonotonic_steps = true;
	init_options.num_linear_solver_threads = 1;
	init_options.minimizer_progress_to_stdout = true;
	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);

	ceres::Solve(init_options, &init_problem, &init_summary);
	std::cout << init_summary.FullReport() << std::endl;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-5,
		ceres::TAKE_OWNERSHIP);
	init_problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-2,
		ceres::TAKE_OWNERSHIP);
	init_problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	init_options.function_tolerance = 1e-4;
	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(init_options, &init_problem, &init_summary);
	std::cout << init_summary.FullReport() << std::endl;
}

void Adam_skeletal_refit(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type,
	bool bFreezeShape,
	bool euler)
{
	Eigen::MatrixXd surface_constraint(6, 0);
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true);  // fit 3D only
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type, false, euler);

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	if (euler)
	{
		ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
			<AdamBodyPoseParamPrior,
			TotalModel::NUM_POSE_PARAMETERS,
			TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			1e-2,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			frame_param.m_adam_pose.data());
	}
	else
	{
		AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
		ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
			1e-2,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(cost_prior_body_pose,
			loss_weight_prior_body_pose,
			frame_param.m_adam_pose.data());
		Eigen::Matrix<double, 72, TotalModel::NUM_POSE_PARAMETERS> prior_A; prior_A.setZero();
		prior_A.block<72, 66>(0, 0) = adam.smpl_pose_prior_A.block<72, 66>(0, 0);  // for body, use the prior from SMPL
		// prior_A.block<6, 6>(13 * 3, 13 * 3) *= 10;
		// prior_A.block<6, 6>(13 * 3, 16 * 3) *= 10;
		// prior_A.block<6, 6>(16 * 3, 16 * 3) *= 10;
		// prior_A.block<6, 6>(16 * 3, 13 * 3) *= 10;
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> prior_mu; prior_mu.setZero();
		prior_mu.block<66, 1>(0, 0) = -adam.smpl_pose_prior_mu.block<66, 1>(0, 0);  // use the prior from SMPL
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(prior_A, prior_mu);
		ceres::LossFunction* pose_reg_loss = new ceres::ScaledLoss(NULL,
			1,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(pose_reg,
			pose_reg_loss,
			frame_param.m_adam_pose.data());
		for (int i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++) cost_prior_body_pose->weight[i] = 0.0;  // only use regularization for fingers
		Eigen::MatrixXd hand_prior_A(120, TotalModel::NUM_POSE_PARAMETERS); hand_prior_A.setZero();
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> hand_prior_mu; hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(66, 0) = -adam.hand_pose_prior_mu; hand_prior_mu.block<60, 1>(126, 0) = adam.hand_pose_prior_mu;
		for (int i = 66; i < 126; i += 3) hand_prior_mu(i, 0) = -hand_prior_mu(i, 0);
		hand_prior_A.block<60, 60>(0, 66) = -adam.hand_pose_prior_A; hand_prior_A.block<60, 60>(60, 126) = adam.hand_pose_prior_A;
		for (int i = 66; i < 126; i += 3) hand_prior_A.col(i) = -hand_prior_A.col(i);
		ceres::CostFunction *hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
		ceres::LossFunction *hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
			0.01,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(hand_pose_reg,
			hand_pose_reg_loss,
			frame_param.m_adam_pose.data());
	}

	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 3] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 6] =
	// adam_cost->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 7] = 1.0;  // foot vertices
	std::fill(adam_cost->m_targetPts_weight.data() + 5 * adam_cost->m_nCorrespond_adam2joints,
		      adam_cost->m_targetPts_weight.data() + 5 * (adam_cost->m_nCorrespond_adam2joints + 8), 1.0);

	if (bFreezeShape)
	{
		problem.SetParameterBlockConstant(frame_param.m_adam_coeffs.data());
	}

	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options, &problem, &summary);
	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	adam_cost->toggle_activate(true, true, true);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// AdamFitData data_new(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true);  // fit 3D only
	// AdamFullCost* adam_cost_new;
	// adam_cost_new = new AdamFullCost(data_new, regressor_type);
	// ceres::Problem problem_new;
	// ceres::Solver::Options options_new;
	// ceres::Solver::Summary summary_new;
	// problem_new.AddResidualBlock(adam_cost_new,
	// 	NULL,
	// 	frame_param.m_adam_t.data(),
	// 	frame_param.m_adam_pose.data(),
	// 	frame_param.m_adam_coeffs.data());

	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 0] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 1] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 2] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 3] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 4] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 5] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 6] =
	// adam_cost_new->m_targetPts_weight[adam_cost->m_nCorrespond_adam2joints + 7] = 1.0;  // foot vertices

	// //Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	// CoeffsParameterNormDiff* cost_prior_body_coeffs_new = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	// ceres::LossFunction* loss_weight_prior_body_coeffs_new = new ceres::ScaledLoss(NULL,
	// 	1e-5,
	// 	ceres::TAKE_OWNERSHIP);
	// problem_new.AddResidualBlock(cost_prior_body_coeffs_new,
	// 	loss_weight_prior_body_coeffs_new,
	// 	frame_param.m_adam_coeffs.data());
	// //Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	// ceres::CostFunction *cost_prior_body_pose_new = new ceres::AutoDiffCostFunction
	// 	<AdamBodyPoseParamPrior,
	// 	TotalModel::NUM_POSE_PARAMETERS,
	// 	TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
	// ceres::LossFunction* loss_weight_prior_body_pose_new = new ceres::ScaledLoss(NULL,
	// 	1e-5,
	// 	ceres::TAKE_OWNERSHIP);
	// problem_new.AddResidualBlock(cost_prior_body_pose_new,
	// 	loss_weight_prior_body_pose_new,
	// 	frame_param.m_adam_pose.data());

	// SetSolverOptions(&options_new);
	// options_new.function_tolerance = 1e-4;
	// options_new.max_num_iterations = 30;
	// options_new.use_nonmonotonic_steps = true;
	// options_new.num_linear_solver_threads = 1;
	// options_new.minimizer_progress_to_stdout = true;
	// adam_cost_new->toggle_activate(true, true, false);
	// ceres::Solve(options_new, &problem_new, &summary_new);
	// std::cout << summary_new.FullReport() << std::endl;
	// adam_cost_new->toggle_activate(true, true, true);
	// ceres::Solve(options_new, &problem_new, &summary_new);
	// std::cout << summary_new.FullReport() << std::endl;
}

void Adam_skeletal_init(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &BodyJoints,
	Eigen::MatrixXd &rFoot,
	Eigen::MatrixXd &lFoot,
	Eigen::MatrixXd &rHandJoints,
	Eigen::MatrixXd &lHandJoints,
	Eigen::MatrixXd &faceJoints,
	Eigen::MatrixXd &PAF,
	uint regressor_type)
{
	std::cout << "3D skeletal refitting" << std::endl;
	Eigen::MatrixXd surface_constraint(6, 0);
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true);  // fit 3D only
	AdamFullCost* adam_cost;
	adam_cost = new AdamFullCost(data, regressor_type);

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;

	adam_cost->toggle_activate(false, false, false);
	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1e-1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());

	adam_cost->toggle_activate(true, false, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// std::fill(adam_cost->m_targetPts_weight.data() + 19, adam_cost->m_targetPts_weight.data() + 59, 5);
	adam_cost->toggle_activate(true, true, false);
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}

void Adam_align_mano(TotalModel &adam,
	smpl::SMPLParams &frame_param,
	Eigen::MatrixXd &surface_constraint)
{
	using namespace Eigen;
	MatrixXd BodyJoints(5, 20); std::fill(BodyJoints.data(), BodyJoints.data() + 100, 0);
	MatrixXd rFoot(5, 3); std::fill(rFoot.data(), rFoot.data() + 15, 0);
	MatrixXd lFoot(5, 3); std::fill(lFoot.data(), lFoot.data() + 15, 0);
	MatrixXd lHandJoints(5, 21); std::fill(lHandJoints.data(), lHandJoints.data() + 105, 0);
	MatrixXd rHandJoints(5, 21); std::fill(rHandJoints.data(), rHandJoints.data() + 105, 0);
	MatrixXd PAF(3, 54); std::fill(PAF.data(), PAF.data() + 108, 0);
	MatrixXd faceJoints(5, 70); std::fill(faceJoints.data(), faceJoints.data() + 350, 0);
	AdamFitData data(adam, BodyJoints, rFoot, lFoot, faceJoints, lHandJoints, rHandJoints, PAF, surface_constraint, true, false, nullptr, false, true);  // fit surface only

	AdamFullCost* adam_cost = new AdamFullCost(data, 2, false, false);  // using angle axis representation
	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	problem.AddResidualBlock(adam_cost,
		NULL,
		frame_param.m_adam_t.data(),
		frame_param.m_adam_pose.data(),
		frame_param.m_adam_coeffs.data());

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = true;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-4,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		frame_param.m_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	AdamBodyPoseParamPriorDiff* cost_prior_body_pose = new AdamBodyPoseParamPriorDiff(TotalModel::NUM_POSE_PARAMETERS);
	ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
		1,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_pose,
		loss_weight_prior_body_pose,
		frame_param.m_adam_pose.data());
	std::fill(cost_prior_body_pose->weight.data() + 3, cost_prior_body_pose->weight.data() + TotalModel::NUM_POSE_PARAMETERS, 1.0);

	std::fill(adam_cost->m_targetPts_weight.data(), adam_cost->m_targetPts_weight.data() + 5 * (8 + adam.m_correspond_adam2face70_adamIdx.rows()), 0.0);
	for (int i = 0; i < surface_constraint.cols(); i++)
		for (auto d = 0; d < 5; d++)
			adam_cost->m_targetPts_weight[5 * (adam_cost->m_nCorrespond_adam2joints + 8 + adam.m_correspond_adam2face70_adamIdx.rows() + i) + d] = 1.0; // surface constraints

	adam_cost->toggle_rigid_body(true);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
	adam_cost->toggle_rigid_body(false);
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}

void Adam_refit_batch(TotalModel &adam,
	std::vector<smpl::SMPLParams*> &frame_param,
	std::vector<Eigen::MatrixXd> &BodyJoints,
	std::vector<Eigen::MatrixXd> &rFoot,
	std::vector<Eigen::MatrixXd> &lFoot,
	std::vector<Eigen::MatrixXd> &rHandJoints,
	std::vector<Eigen::MatrixXd> &lHandJoints,
	std::vector<Eigen::MatrixXd> &faceJoints,
	std::vector<Eigen::MatrixXd> &PAF,
	uint regressor_type,
	bool bFreezeShape,
	bool euler,
	bool bDCT)
{
	const uint num_t = frame_param.size();
	Eigen::MatrixXd surface_constraint(6, 0);

	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;

	auto& common_adam_coeffs = frame_param[0]->m_adam_coeffs;

	std::vector<AdamFitData> data;
	std::vector<AdamFullCost*> adam_cost;
	data.reserve(num_t);  // The capacity must be remained. Otherwise previous reference (in AdamFullCost) will be invalidated.

	for (uint t = 0u; t < num_t; t++)
	{
		data.emplace_back(adam, BodyJoints[t], rFoot[t], lFoot[t], faceJoints[t], lHandJoints[t], rHandJoints[t], PAF[t], surface_constraint, true);
		adam_cost.emplace_back(new AdamFullCost(data.back(), regressor_type, false, euler));
		problem.AddResidualBlock(adam_cost.back(),
			NULL,
			frame_param[t]->m_adam_t.data(),
			frame_param[t]->m_adam_pose.data(),
			common_adam_coeffs.data());
	}

	SetSolverOptions(&options);
	options.function_tolerance = 1e-4;
	options.max_num_iterations = 30;
	options.use_nonmonotonic_steps = false;
	options.num_linear_solver_threads = 1;
	options.minimizer_progress_to_stdout = true;

	//Body Prior (coef) //////////////////////////////////////////////////////////////////////////
	CoeffsParameterNormDiff* cost_prior_body_coeffs = new CoeffsParameterNormDiff(TotalModel::NUM_SHAPE_COEFFICIENTS);
	ceres::LossFunction* loss_weight_prior_body_coeffs = new ceres::ScaledLoss(NULL,
		1e-4 * num_t,
		ceres::TAKE_OWNERSHIP);
	problem.AddResidualBlock(cost_prior_body_coeffs,
		loss_weight_prior_body_coeffs,
		common_adam_coeffs.data());

	//Body Prior (pose) //////////////////////////////////////////////////////////////////////////
	if (euler)
	{
		for (uint t = 0u; t < num_t; t++)
		{
			ceres::CostFunction *cost_prior_body_pose = new ceres::AutoDiffCostFunction
				<AdamBodyPoseParamPrior,
				TotalModel::NUM_POSE_PARAMETERS,
				TotalModel::NUM_POSE_PARAMETERS>(new AdamBodyPoseParamPrior(TotalModel::NUM_POSE_PARAMETERS));
			ceres::LossFunction* loss_weight_prior_body_pose = new ceres::ScaledLoss(NULL,
				1e-2,
				ceres::TAKE_OWNERSHIP);
			problem.AddResidualBlock(cost_prior_body_pose,
				loss_weight_prior_body_pose,
				frame_param[t]->m_adam_pose.data());
		}
	}
	else
	{
		Eigen::Matrix<double, 72, TotalModel::NUM_POSE_PARAMETERS> prior_A; prior_A.setZero();
		prior_A.block<72, 66>(0, 0) = adam.smpl_pose_prior_A.block<72, 66>(0, 0);  // for body, use the prior from SMPL
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> prior_mu; prior_mu.setZero();
		prior_mu.block<66, 1>(0, 0) = -adam.smpl_pose_prior_mu.block<66, 1>(0, 0);  // use the prior from SMPL

		Eigen::MatrixXd hand_prior_A(120, TotalModel::NUM_POSE_PARAMETERS); hand_prior_A.setZero();
		Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> hand_prior_mu; hand_prior_mu.setZero();
		hand_prior_mu.block<60, 1>(66, 0) = -adam.hand_pose_prior_mu; hand_prior_mu.block<60, 1>(126, 0) = adam.hand_pose_prior_mu;
		for (int i = 66; i < 126; i += 3) hand_prior_mu(i, 0) = -hand_prior_mu(i, 0);
		hand_prior_A.block<60, 60>(0, 66) = -adam.hand_pose_prior_A; hand_prior_A.block<60, 60>(60, 126) = adam.hand_pose_prior_A;
		for (int i = 66; i < 126; i += 3) hand_prior_A.col(i) = -hand_prior_A.col(i);

		for (uint t = 0u; t < num_t; t++)
		{
			ceres::CostFunction *pose_reg = new ceres::NormalPrior(prior_A, prior_mu);
			ceres::LossFunction* pose_reg_loss = new ceres::ScaledLoss(NULL,
				1,
				ceres::TAKE_OWNERSHIP);
			problem.AddResidualBlock(pose_reg,
				pose_reg_loss,
				frame_param[t]->m_adam_pose.data());
			ceres::CostFunction *hand_pose_reg = new ceres::NormalPrior(hand_prior_A, hand_prior_mu);
			ceres::LossFunction *hand_pose_reg_loss = new ceres::ScaledLoss(NULL,
				0.1,
				ceres::TAKE_OWNERSHIP);
			problem.AddResidualBlock(hand_pose_reg,
				hand_pose_reg_loss,
				frame_param[t]->m_adam_pose.data());
		}
	}

	// DCT smoothing loss
	std::vector<double*> pose_blocks;
	for (uint t = 0u; t < num_t; t++) pose_blocks.emplace_back(frame_param[t]->m_adam_pose.data());
	std::vector<double*> trans_blocks;
	for (uint t = 0u; t < num_t; t++) trans_blocks.emplace_back(frame_param[t]->m_adam_t.data());

	const uint low_comp = ceil(0.125 * num_t);  // should not be 0
	const uint low_comp_hand = ceil(0.125 * num_t);
	if (bDCT)
	{
		DCTCost* smooth_cost_pose = new DCTCost(num_t, low_comp, TotalModel::NUM_POSE_PARAMETERS, 0, 60, 500);  // DCT for body only
		problem.AddResidualBlock(smooth_cost_pose, nullptr, pose_blocks);
		DCTCost* smooth_cost_wrist = new DCTCost(num_t, low_comp, TotalModel::NUM_POSE_PARAMETERS, 60, 66, 100);  // DCT for wrist only
		problem.AddResidualBlock(smooth_cost_wrist, nullptr, pose_blocks);
		DCTCost* smooth_cost_pose_hand = new DCTCost(num_t, low_comp_hand, TotalModel::NUM_POSE_PARAMETERS, 66, 186, 10);  // DCT for fingers only
		problem.AddResidualBlock(smooth_cost_pose_hand, nullptr, pose_blocks);
		DCTCost* smooth_cost_trans = new DCTCost(num_t, low_comp, 3, 0, 3, 10);
		problem.AddResidualBlock(smooth_cost_trans, nullptr, trans_blocks);
	}
	else
	{
		// smoothing on body is too strong, do not fit the fingers
		/*
		for (uint t = 0u; t < num_t; t++)
		{
			Eigen::MatrixXd ones = Eigen::MatrixXd::Zero(TotalModel::NUM_POSE_PARAMETERS - 60, TotalModel::NUM_POSE_PARAMETERS);
			ones.block(0, 60, TotalModel::NUM_POSE_PARAMETERS - 60, TotalModel::NUM_POSE_PARAMETERS - 60).setIdentity();
			Eigen::VectorXd b(TotalModel::NUM_POSE_PARAMETERS);
			std::copy(frame_param[t]->m_adam_pose.data(), frame_param[t]->m_adam_pose.data() + TotalModel::NUM_POSE_PARAMETERS, b.data());
			ceres::CostFunction* keep_hand_joint_angle = new ceres::NormalPrior(ones, b);
			ceres::LossFunction* keep_hand_joint_angle_loss = new ceres::ScaledLoss(NULL,
				100000,
			ceres::TAKE_OWNERSHIP);
			problem.AddResidualBlock(keep_hand_joint_angle, keep_hand_joint_angle_loss, frame_param[t]->m_adam_pose.data());
		}
		*/
		TemporalSmoothCostDiff* smooth_cost_pose = new TemporalSmoothCostDiff(num_t, TotalModel::NUM_POSE_PARAMETERS, 0, 60);
		ceres::LossFunction *smooth_cost_pose_loss = new ceres::ScaledLoss(NULL,
			250000,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(smooth_cost_pose, smooth_cost_pose_loss, pose_blocks);
		TemporalSmoothCostDiff* smooth_cost_pose_wrist = new TemporalSmoothCostDiff(num_t, TotalModel::NUM_POSE_PARAMETERS, 60, 66);
		ceres::LossFunction *smooth_cost_pose_loss_wrist = new ceres::ScaledLoss(NULL,
			10000,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(smooth_cost_pose_wrist, smooth_cost_pose_loss_wrist, pose_blocks);
		TemporalSmoothCostDiff* smooth_cost_pose_hand = new TemporalSmoothCostDiff(num_t, TotalModel::NUM_POSE_PARAMETERS, 66, TotalModel::NUM_POSE_PARAMETERS);
		ceres::LossFunction *smooth_cost_pose_loss_hand = new ceres::ScaledLoss(NULL,
			100,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(smooth_cost_pose_hand, smooth_cost_pose_loss_hand, pose_blocks);
		TemporalSmoothCostDiff* smooth_cost_trans = new TemporalSmoothCostDiff(num_t, 3, 0, 3);
		ceres::LossFunction *smooth_cost_trans_loss = new ceres::ScaledLoss(NULL,
			500,
			ceres::TAKE_OWNERSHIP);
		problem.AddResidualBlock(smooth_cost_trans, smooth_cost_trans_loss, trans_blocks);
	}

	for (uint t = 0; t < num_t; t++)
	{
		std::fill(adam_cost[t]->m_targetPts_weight.data() + 5 * adam_cost[t]->m_nCorrespond_adam2joints,
		      	  adam_cost[t]->m_targetPts_weight.data() + 5 * (adam_cost[t]->m_nCorrespond_adam2joints + 8), 1.0);
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 0] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 1] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 2] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 3] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 4] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 5] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 6] =
		// adam_cost[t]->m_targetPts_weight[adam_cost[t]->m_nCorrespond_adam2joints + 7] = 1.0;  // foot vertices
	}

	if (bFreezeShape)
	{
		problem.SetParameterBlockConstant(common_adam_coeffs.data());
	}

	// for (uint t = 0; t < num_t; t++)
	// {
	// 	adam_cost[t]->toggle_activate(false, false, false);
	// 	adam_cost[t]->toggle_rigid_body(true);
	// }
	// ceres::Solve(options, &problem, &summary);
	// for (uint t = 0; t < num_t; t++)
	// {
	// 	adam_cost[t]->toggle_activate(true, false, false);
	// 	adam_cost[t]->toggle_rigid_body(false);
	// }
	// ceres::Solve(options, &problem, &summary);
	// std::cout << summary.FullReport() << std::endl;

	for (uint t = 0; t < num_t; t++)
	{
		adam_cost[t]->toggle_activate(true, true, true);
		if (!bDCT)
		{
			std::fill(adam_cost[t]->m_targetPts_weight.data() + 6 * 5, adam_cost[t]->m_targetPts_weight.data() + 7 * 5, 1.0);
			std::fill(adam_cost[t]->m_targetPts_weight.data() + 11 * 5, adam_cost[t]->m_targetPts_weight.data() + 12 * 5, 1.0);
			// adam_cost[t]->m_targetPts_weight[6] = adam_cost[t]->m_targetPts_weight[11] = 1;
		}
	}
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// duplicate the shape coeffs
	for (uint t = 1; t < num_t; t++)
	{
		std::copy(common_adam_coeffs.data(), common_adam_coeffs.data() + TotalModel::NUM_SHAPE_COEFFICIENTS, frame_param[t]->m_adam_coeffs.data());
	}
}