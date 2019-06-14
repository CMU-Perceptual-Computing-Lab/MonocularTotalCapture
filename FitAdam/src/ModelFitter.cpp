#include <ModelFitter.h>
#include <FitToBody.h>

const int ModelFitter::NUM_KEYPOINTS_2D;
const int ModelFitter::NUM_PAF_VEC;
const uint ModelFitter::num_body_joint;
const uint ModelFitter::num_hand_joint;
const uint ModelFitter::num_face_landmark;

ModelFitter::ModelFitter(const TotalModel& adam): poseRegularizer(TotalModel::NUM_POSE_PARAMETERS),
    regressor_type(2), fit_face_exp(false), euler(true),
    fit3D(false), fit2D(false), fitPAF(false), fit_surface(false), freezeShape(false), shareCoeff(true),
    DCT_trans_start(0), DCT_trans_end(0), DCT_trans_low_comp(0),
    bodyJoints(1, Eigen::MatrixXd::Zero(5, 21)),
    rFoot(1, Eigen::MatrixXd::Zero(5, 3)),
    lFoot(1, Eigen::MatrixXd::Zero(5, 3)),
    faceJoints(1, Eigen::MatrixXd::Zero(5, 70)),
    lHandJoints(1, Eigen::MatrixXd::Zero(5, 21)),
    rHandJoints(1, Eigen::MatrixXd::Zero(5, 21)),
    PAF(1, Eigen::MatrixXd::Zero(3, 63)),
    surface_constraint(1, Eigen::MatrixXd::Zero(6, 0)),
    wPoseRg(1e-2), wCoeffRg(1e-2), wPosePr(10), wHandPosePr(10), wFacePr(100),
    adam(adam), costFunctionInit(false), calibKInit(false), fitDataInit(false),
    pose(1, Eigen::Matrix<double, 62, 3, Eigen::RowMajor>::Zero()), coeffs(1, Eigen::Matrix<double, 30, 1>::Zero()), trans(1, Eigen::Vector3d::Zero()), exp(1, Eigen::Matrix<double, 200, 1>::Zero()),
    hand_prior_A(120, TotalModel::NUM_POSE_PARAMETERS), coeffRegularizer(TotalModel::NUM_SHAPE_COEFFICIENTS), pPosePriorCost(nullptr), pHandPosePriorCost(nullptr),
    facePriorCost(adam.face_prior_A_exp.asDiagonal(), Eigen::Matrix<double, TotalModel::NUM_EXP_BASIS_COEFFICIENTS, 1>::Zero()), timeStep(1),
    smooth_cost_trans(nullptr)
{
    std::fill(calibK.data(), calibK.data() + 9, 0.0);

    // default ceres options
    options.function_tolerance = 1e-4;
    options.parameter_tolerance = 1e-15;
    options.max_num_iterations = 30;
    options.use_nonmonotonic_steps = true;
    options.num_linear_solver_threads = 1;
    options.minimizer_progress_to_stdout = true;

    // use the prior from SMPL, negative because we are use inverse angle axis
    prior_A.setZero();
    prior_A.block<72, 66>(0, 0) = adam.smpl_pose_prior_A.block<72, 66>(0, 0);
    prior_mu.setZero();
    prior_mu.block<66, 1>(0, 0) = -adam.smpl_pose_prior_mu.block<66, 1>(0, 0);

    // the hand prior is build from MANO for right hand, set value for left hand as well
    hand_prior_A.setZero();
    hand_prior_mu.setZero();
    hand_prior_mu.block<60, 1>(66, 0) = -adam.hand_pose_prior_mu; hand_prior_mu.block<60, 1>(126, 0) = adam.hand_pose_prior_mu;
    for (int i = 66; i < 126; i += 3) hand_prior_mu(i, 0) = -hand_prior_mu(i, 0);
    hand_prior_A.block<60, 60>(0, 66) = -adam.hand_pose_prior_A; hand_prior_A.block<60, 60>(60, 126) = adam.hand_pose_prior_A;
    for (int i = 66; i < 126; i += 3) hand_prior_A.col(i) = -hand_prior_A.col(i);

    pPosePriorCost.reset(new ceres::NormalPrior(prior_A, prior_mu));
    pHandPosePriorCost.reset(new ceres::NormalPrior(hand_prior_A, hand_prior_mu));
}

void ModelFitter::resetTimeStep(const uint t)
{
    assert(t > 0);
    this->timeStep = t;
    bodyJoints.resize(t, Eigen::MatrixXd::Zero(5, 21));
    rFoot.resize(t, Eigen::MatrixXd::Zero(5, 3));
    lFoot.resize(t, Eigen::MatrixXd::Zero(5, 3));
    lHandJoints.resize(t, Eigen::MatrixXd::Zero(5, 21));
    rHandJoints.resize(t, Eigen::MatrixXd::Zero(5, 21));
    PAF.resize(t, Eigen::MatrixXd::Zero(3, 63));
    faceJoints.resize(t, Eigen::MatrixXd::Zero(5, 70));
    surface_constraint.resize(t, Eigen::MatrixXd::Zero(6, 0));
    pose.resize(t, Eigen::Matrix<double, 62, 3, Eigen::RowMajor>::Zero());
    coeffs.resize(t, Eigen::Matrix<double, 30, 1>::Zero());
    trans.resize(t, Eigen::Vector3d::Zero());
    exp.resize(t, Eigen::Matrix<double, 200, 1>::Zero());
}

void ModelFitter::initParameters(const smpl::SMPLParams& frame_params, const uint t)
{
    assert(t < timeStep);
    std::copy(frame_params.m_adam_pose.data(), frame_params.m_adam_pose.data() + 62 * 3, pose[t].data());
    std::copy(frame_params.m_adam_coeffs.data(), frame_params.m_adam_coeffs.data() + 30, coeffs[t].data());
    std::copy(frame_params.m_adam_t.data(), frame_params.m_adam_t.data() + 3, trans[t].data());
    std::copy(frame_params.m_adam_facecoeffs_exp.data(), frame_params.m_adam_facecoeffs_exp.data() + 200, exp[t].data());
}

void ModelFitter::readOutParameters(smpl::SMPLParams& frame_params, const uint t)
{
    std::copy(pose[t].data(), pose[t].data() + 62 * 3, frame_params.m_adam_pose.data());
    std::copy(coeffs[t].data(), coeffs[t].data() + 30, frame_params.m_adam_coeffs.data());
    std::copy(trans[t].data(), trans[t].data() + 3, frame_params.m_adam_t.data());
    std::copy(exp[t].data(), exp[t].data() + 200, frame_params.m_adam_facecoeffs_exp.data());
}

void ModelFitter::setCalibK(const double* K)
{
    assert(K);
    std::copy(K, K + 9, this->calibK.data());
    this->calibKInit = true;
}

void ModelFitter::resetFitData()
{
    this->checkTimeStepEqual();
    if (fit2D) assert(this->calibKInit);  // if fit 2D, camera intrinsics must be provided
    this->FitData.clear();
    for (uint t = 0u; t < timeStep; t++)
        this->FitData.emplace_back(this->adam, bodyJoints[t], rFoot[t], lFoot[t], faceJoints[t], lHandJoints[t], rHandJoints[t], PAF[t], surface_constraint[t],
                                                        fit3D, fit2D, this->calibK.data(), fitPAF, fit_surface);
    this->fitDataInit = true;
    std::cout << "Create AdamFitData: " << std::endl
              << "Fit 3D: " << fit3D << std::endl
              << "Fit 2D: " << fit2D << std::endl
              << "Fit PAF: " << fitPAF << std::endl
              << "Fit Surface: " << fit_surface << std::endl;
}

void ModelFitter::resetCostFunction()
{
    if (!this->fitDataInit) resetFitData();
    assert(this->fitDataInit);
    std::cout << "Create AdamFullCost" << std::endl
              << "Regressor type: " << regressor_type << std::endl
              << "Fit facial expression: " << fit_face_exp << std::endl
              << "Using Euler angles: " << euler << std::endl;
    this->checkTimeStepEqual();
    assert(FitData.size() == timeStep);
    this->pCostFunction.clear();
    for (uint t = 0u; t < timeStep; t++)
        this->pCostFunction.emplace_back(std::make_shared<AdamFullCost>(this->FitData[t], regressor_type, fit_face_exp, euler));
    this->costFunctionInit = true;
}

void ModelFitter::runFitting()
{
    if (!this->costFunctionInit) resetCostFunction();
    assert(this->costFunctionInit);
    assert(pCostFunction.size() == timeStep);
    ceres::Problem::Options problem_op;
    problem_op.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;  // do not destroy AdamFullCost
    ceres::Problem problem(problem_op);

    for (uint t = 0u; t < this->timeStep; t++)
    {
        Eigen::Matrix<double, 30, 1>& fit_coeffs = (this->shareCoeff) ? coeffs[0] : coeffs[t];
        if (this->fit_face_exp)
            problem.AddResidualBlock(this->pCostFunction[t].get(),  // problem does not take the ownership of cost_function, can use get here
                nullptr,  
                trans[t].data(),
                pose[t].data(),
                fit_coeffs.data(),
                exp[t].data());
        else
            problem.AddResidualBlock(this->pCostFunction[t].get(),  // problem does not take the ownership of cost_function, can use get here
                nullptr, 
                trans[t].data(),
                pose[t].data(),
                fit_coeffs.data());

        ceres::LossFunction* lossCoeffRegularizer = new ceres::ScaledLoss(nullptr, wCoeffRg, ceres::TAKE_OWNERSHIP);  // take ownership of 'nullptr'
        problem.AddResidualBlock(&coeffRegularizer, lossCoeffRegularizer, fit_coeffs.data());

        if (this->euler)   // use regularizers for pose
        {
            ceres::LossFunction* lossPoseRegularizer = new ceres::ScaledLoss(nullptr, wPoseRg, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(&poseRegularizer, lossPoseRegularizer, pose[t].data());
        }
        else  // use pose prior
        {
            // need this regularizer to prevent the bending in abdomen
            // for (uint i = 0; i < TotalModel::NUM_POSE_PARAMETERS; i++)
            // {
            //     if ((i >= 9 && i < 12) || (i >= 18 && i < 21) || (i >= 27 && i < 30))
            //         poseRegularizer.weight[i] = 50;
            //     else
            //         poseRegularizer.weight[i] = 0.0;
            // }
            // ceres::LossFunction* lossPoseRegularizer = new ceres::ScaledLoss(nullptr, 1, ceres::TAKE_OWNERSHIP);
            // problem.AddResidualBlock(&poseRegularizer, lossPoseRegularizer, pose[t].data());
            ceres::LossFunction* lossPosePriorCost = new ceres::ScaledLoss(nullptr, wPosePr, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(pPosePriorCost.get(), lossPosePriorCost, pose[t].data());
            ceres::LossFunction* lossHandPosePriorCost = new ceres::ScaledLoss(nullptr, wHandPosePr, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(pHandPosePriorCost.get(), lossHandPosePriorCost, pose[t].data());
        }
        if (this->fit_face_exp)
        {
            ceres::LossFunction* lossFacePriorCost = new ceres::ScaledLoss(nullptr, wFacePr, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(&facePriorCost, lossFacePriorCost, exp[t].data());
        }

        if (this->freezeShape)
            problem.SetParameterBlockConstant(fit_coeffs.data());
    }

    if (DCT_trans_start < DCT_trans_end)
    {
        assert(DCT_trans_end <= 3);
        assert(DCT_trans_low_comp > 0 && DCT_trans_low_comp < timeStep);
        // DCT smoothing on translation is triggered
        std::vector<double*> trans_blocks;
        for (uint t = 0u; t < timeStep; t++) trans_blocks.emplace_back(trans[t].data());
        assert(!smooth_cost_trans);  // should keep a nullptr
        smooth_cost_trans = new DCTCost(timeStep, DCT_trans_low_comp, 3, DCT_trans_start, DCT_trans_end, 500);  // only regularize the z-direction
        problem.AddResidualBlock(smooth_cost_trans, nullptr, trans_blocks);
    }

    // run solver now
    ceres::Solve(options, &problem, &summary);
    std::cout << this->summary.FullReport() << std::endl;
    if (DCT_trans_start < DCT_trans_end)
    {
        delete smooth_cost_trans;
        smooth_cost_trans = nullptr;
    }

    if (this->freezeShape)
    {
        if (this->shareCoeff)
            problem.SetParameterBlockVariable(coeffs[0].data());
        else
        {
            for (uint t = 0u; t < timeStep; t++)
            {
                problem.SetParameterBlockVariable(coeffs[t].data());
            }
        }
    }

    if (shareCoeff)
    {
        for (uint t = 1u; t < timeStep; t++)
            coeffs[t] = coeffs[0];
    }
}

void ModelFitter::setSurfaceConstraints2D(const std::vector<cv::Point3i>& surfaceConstraint2D, const uint t)
{
    assert(t < timeStep);
    if ((uint)surface_constraint[t].cols() != surfaceConstraint2D.size())
        surface_constraint[t].resize(6, surfaceConstraint2D.size());
    surface_constraint[t].block(0, 0, 3, surfaceConstraint2D.size()).setZero();
    for (auto i = 0u; i < surfaceConstraint2D.size(); i++)
    {
        surface_constraint[t](3, i) = surfaceConstraint2D[i].x;
        surface_constraint[t](4, i) = surfaceConstraint2D[i].y;
        surface_constraint[t](5, i) = surfaceConstraint2D[i].z;
    }
}

void ModelFitter::setFitDataNetOutput(
    const std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2> net_output,
    const bool copyBody,
    const bool copyFoot,
    const bool copyHand,
    const bool copyFace,
    const bool copyPAF,
    const uint t
)
{
    assert(t < timeStep);
    if (copyBody)
    {
        for (uint i = 0; i < num_body_joint; i++)
        {
            // this->bodyJoints[t](0, i) = 0;
            // this->bodyJoints[t](1, i) = 0;
            // this->bodyJoints[t](2, i) = 0;
            this->bodyJoints[t](3, i) = net_output[2 * i];
            this->bodyJoints[t](4, i) = net_output[2 * i + 1];
        }
    }
    if (copyHand)
    {
        for (uint i = 0; i < num_hand_joint; i++)
        {
            // this->lHandJoints[t](0, i) = 0;
            // this->lHandJoints[t](1, i) = 0;
            // this->lHandJoints[t](2, i) = 0;
            this->lHandJoints[t](3, i) = net_output[2 * (i + num_body_joint)];
            this->lHandJoints[t](4, i) = net_output[2 * (i + num_body_joint) + 1];
        }
        for (uint i = 0; i < num_hand_joint; i++)
        {
            // this->rHandJoints[t](0, i) = 0;
            // this->rHandJoints[t](1, i) = 0;
            // this->rHandJoints[t](2, i) = 0;
            this->rHandJoints[t](3, i) = net_output[2 * (i + num_hand_joint + num_body_joint)];
            this->rHandJoints[t](4, i) = net_output[2 * (i + num_hand_joint + num_body_joint) + 1];
        }
    }
    if (copyFace)
    {
        for (uint i = 0; i < num_face_landmark; i++)
        {
            // this->faceJoints[t](0, i) = 0;
            // this->faceJoints[t](1, i) = 0;
            // this->faceJoints[t](2, i) = 0;
            this->faceJoints[t](3, i) = net_output[2 * (i + num_hand_joint + num_body_joint + num_hand_joint)];
            this->faceJoints[t](4, i) = net_output[2 * (i + num_hand_joint + num_body_joint + num_hand_joint) + 1];
        }
    }
    if (copyPAF)
    {
        const double* PAF_vec = net_output.data() + 2 * ModelFitter::NUM_KEYPOINTS_2D; 
        for (int i = 0; i < ModelFitter::NUM_PAF_VEC; i++)
        {
            if (PAF_vec[3 * i + 0] != 0.0 || PAF_vec[3 * i + 1] != 0.0 || PAF_vec[3 * i + 2] != 0.0)
            {
                const auto length = sqrt(PAF_vec[3 * i + 0] * PAF_vec[3 * i + 0] + PAF_vec[3 * i + 1] * PAF_vec[3 * i + 1] + PAF_vec[3 * i + 2] * PAF_vec[3 * i + 2]);
                this->PAF[t](0, i) = PAF_vec[3 * i + 0] / length;
                this->PAF[t](1, i) = PAF_vec[3 * i + 1] / length;
                this->PAF[t](2, i) = PAF_vec[3 * i + 2] / length;
            }
            else this->PAF[t].col(i).setZero();
        }
    }
    if (copyFoot)
    {
        // lFoot[t].setZero(); rFoot[t].setZero();
        this->lFoot[t](3, 0) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0];  // Left BigToe
        this->lFoot[t](4, 0) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1];  // Left BigToe
        this->lFoot[t](3, 1) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2];  // Left SmallToe
        this->lFoot[t](4, 1) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3];  // Left SmallToe
        this->lFoot[t](3, 2) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4];  // Left Heel
        this->lFoot[t](4, 2) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5];  // Left Heel
        this->rFoot[t](3, 0) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0 + 6];  // Right BigToe
        this->rFoot[t](4, 0) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1 + 6];  // Right BigToe
        this->rFoot[t](3, 1) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2 + 6];  // Right SmallToe
        this->rFoot[t](4, 1) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3 + 6];  // Right SmallToe
        this->rFoot[t](3, 2) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4 + 6];  // Right Heel
        this->rFoot[t](4, 2) = net_output[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5 + 6];  // Right Heel
    }
}

void ModelFitter::checkTimeStepEqual()
{
    assert(timeStep == bodyJoints.size());
    assert(timeStep == rFoot.size());
    assert(timeStep == lFoot.size());
    assert(timeStep == faceJoints.size());
    assert(timeStep == PAF.size());
    assert(timeStep == lHandJoints.size());
    assert(timeStep == rHandJoints.size());
    assert(timeStep == surface_constraint.size());

    assert(timeStep == pose.size());
    assert(timeStep == coeffs.size());
    assert(timeStep == trans.size());
    assert(timeStep == exp.size());
}

void ModelFitter::setFitDataReconstruction(const std::vector<double>& reconstruction, const uint t)
{
    // Set fitting target from reconstructed Adam model. Only 3D joint keypoints
    // The mapping from cocoplus regressor to smc order
    const std::array<int, num_body_joint> cocoplus_to_smc = {{12, 14, 12, 9, 10, 11, 3, 4, 5, 8, 7, 6, 2, 1, 0, 15, 17, 16, 18, 13, 19}};
    // set the data
    for (uint i = 0; i < num_body_joint; i++)
    {
        bodyJoints[t](0, i) = reconstruction[3 * cocoplus_to_smc[i] + 0];
        bodyJoints[t](1, i) = reconstruction[3 * cocoplus_to_smc[i] + 1];
        bodyJoints[t](2, i) = reconstruction[3 * cocoplus_to_smc[i] + 2];
    }
    bodyJoints[t].block<3, 1>(0, 2) = (bodyJoints[t].block<3, 1>(0, 6) + bodyJoints[t].block<3, 1>(0, 12)) / 2;  // central hip

    int offset = 20; // hand starts here in reconstruction
    lHandJoints[t](0, 0) = reconstruction[3 * 11 + 0];
    lHandJoints[t](1, 0) = reconstruction[3 * 11 + 1];
    lHandJoints[t](2, 0) = reconstruction[3 * 11 + 2];
    for (uint i = 1; i < num_hand_joint; i++)
    {
        lHandJoints[t](0, i) = reconstruction[3 * (i - 1 + offset) + 0];
        lHandJoints[t](1, i) = reconstruction[3 * (i - 1 + offset) + 1];
        lHandJoints[t](2, i) = reconstruction[3 * (i - 1 + offset) + 2];
    }

    offset = 20 + 20; // hand starts here in reconstruction
    rHandJoints[t](0, 0) = reconstruction[3 * 6 + 0];
    rHandJoints[t](1, 0) = reconstruction[3 * 6 + 1];
    rHandJoints[t](2, 0) = reconstruction[3 * 6 + 2];
    for (uint i = 1; i < num_hand_joint; i++)
    {
        rHandJoints[t](0, i) = reconstruction[3 * (i - 1 + offset) + 0];
        rHandJoints[t](1, i) = reconstruction[3 * (i - 1 + offset) + 1];
        rHandJoints[t](2, i) = reconstruction[3 * (i - 1 + offset) + 2];
    }

    offset = 20 + 20 + 20;
    rFoot[t](0, 0) = reconstruction[3 * offset + 0];  // Right BigToe
    rFoot[t](1, 0) = reconstruction[3 * offset + 1];  // Right BigToe
    rFoot[t](2, 0) = reconstruction[3 * offset + 2];  // Right BigToe
    rFoot[t](0, 1) = reconstruction[3 * offset + 3];  // Right SmallToe
    rFoot[t](1, 1) = reconstruction[3 * offset + 4];  // Right SmallToe
    rFoot[t](2, 1) = reconstruction[3 * offset + 5];  // Right SmallToe
    rFoot[t](0, 2) = reconstruction[3 * offset + 6];  // Right Heel
    rFoot[t](1, 2) = reconstruction[3 * offset + 7];  // Right Heel
    rFoot[t](2, 2) = reconstruction[3 * offset + 8];  // Right Heel

    offset = 20 + 20 + 20 + 3;
    lFoot[t](0, 0) = reconstruction[3 * offset + 0];  // Left BigToe
    lFoot[t](1, 0) = reconstruction[3 * offset + 1];  // Left BigToe
    lFoot[t](2, 0) = reconstruction[3 * offset + 2];  // Left BigToe
    lFoot[t](0, 1) = reconstruction[3 * offset + 3];  // Left SmallToe
    lFoot[t](1, 1) = reconstruction[3 * offset + 4];  // Left SmallToe
    lFoot[t](2, 1) = reconstruction[3 * offset + 5];  // Left SmallToe
    lFoot[t](0, 2) = reconstruction[3 * offset + 6];  // Left Heel
    lFoot[t](1, 2) = reconstruction[3 * offset + 7];  // Left Heel
    lFoot[t](2, 2) = reconstruction[3 * offset + 8];  // Left Heel
}

void ModelFitter::multiStageFitting()
{
    assert(timeStep == 1u);
    pose[0].setZero();
    trans[0].setZero();
    coeffs[0].setZero();
    exp[0].setZero();
    this->fit_surface = false;
    // #1, only fit PAF
    this->fit3D = false;
    this->fit2D = false;
    this->fitPAF = true;
    resetFitData();
    resetCostFunction();
    // pCostFunction[0]->toggle_activate(false, false, false);
    // pCostFunction[0]->toggle_rigid_body(true);
    // runFitting();
    pCostFunction[0]->toggle_activate(true, true, true);
    pCostFunction[0]->toggle_rigid_body(false);
    runFitting();
    trans[0].data()[2] = 200.0;  // set to a good location
    // #2, fit 2D + PAF
    this->fit2D = true;
    resetFitData();
    resetCostFunction();
    if (this->regressor_type == 2)
    {
        std::fill(pCostFunction[0]->m_targetPts_weight.data() + 5 * pCostFunction[0]->m_nCorrespond_adam2joints,
                  pCostFunction[0]->m_targetPts_weight.data() + 5 * (pCostFunction[0]->m_nCorrespond_adam2joints + 8 + adam.m_correspond_adam2face70_adamIdx.rows()), 1.0);
        std::fill(pCostFunction[0]->m_targetPts_weight.data() + 5 * (pCostFunction[0]->m_nCorrespond_adam2joints + 8 + adam.m_correspond_adam2face70_adamIdx.rows()),
                  pCostFunction[0]->m_targetPts_weight.data() + 5 * (pCostFunction[0]->m_nCorrespond_adam2joints + pCostFunction[0]->m_nCorrespond_adam2pts), 0.1);
    }
    pCostFunction[0]->toggle_activate(false, false, false);
    pCostFunction[0]->toggle_rigid_body(true);
    runFitting();
    pCostFunction[0]->toggle_activate(true, false, false);
    pCostFunction[0]->toggle_rigid_body(false);
    runFitting();
    pCostFunction[0]->toggle_activate(true, true, true);
    pCostFunction[0]->toggle_rigid_body(false);
    runFitting();
    if (this->fit_face_exp && regressor_type == 2)
    {
        this->wPosePr *= 0.1;
        pCostFunction[0]->PAF_weight[12] = 5;
        pCostFunction[0]->PAF_weight[13] = 5;
        std::fill(pCostFunction[0]->m_targetPts_weight.data() + 17 * 5, pCostFunction[0]->m_targetPts_weight.data() + 19 * 5, 0.1);  // lower cost for ear, satisfy the face70 constraints
        const Eigen::Matrix<double, 62, 3, Eigen::RowMajor> current_pose = pose[0];
        const Eigen::Matrix<double, 30, 1> current_coeffs = coeffs[0];
        const Eigen::Vector3d current_trans = trans[0];
        runFitting();  // accurately fit the facial expression
        pose[0] = current_pose;
        coeffs[0] = current_coeffs;
        trans[0] = current_trans;
        this->wPosePr *= 10;
    }
}

void fit_single_frame(TotalModel& adam, double* fit_input, double calibK[], smpl::SMPLParams& frame_params, std::vector<std::vector<cv::Point3i>>::iterator densepose_result, bool fit_surface)
{
    const uint num_body_joint = 21;
    const uint num_hand_joint = 21;
    const uint num_face_landmark = 70;
    Eigen::MatrixXd bodyJoints2d(5, num_body_joint);// (3, targetJoint.size());
    Eigen::MatrixXd Joints2d_face(5, num_face_landmark);// (3, landmarks_face.size());
    Eigen::MatrixXd LHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
    Eigen::MatrixXd RHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
    Eigen::MatrixXd LFootJoints2d(5, 3);// (3, 3);      //Heel, Toe
    Eigen::MatrixXd RFootJoints2d(5, 3);// (3, 3);      //Heel, Toe
    Eigen::MatrixXd PAF(3, ModelFitter::NUM_PAF_VEC);
    for (uint i = 0; i < num_body_joint; i++)
    {
        bodyJoints2d(0, i) = 0;
        bodyJoints2d(1, i) = 0;
        bodyJoints2d(2, i) = 0;
        bodyJoints2d(3, i) = fit_input[2 * i];
        bodyJoints2d(4, i) = fit_input[2 * i + 1];
    }
    for (uint i = 0; i < num_hand_joint; i++)
    {
        LHandJoints2d(0, i) = 0;
        LHandJoints2d(1, i) = 0;
        LHandJoints2d(2, i) = 0;
        LHandJoints2d(3, i) = fit_input[2 * (i + num_body_joint)];
        LHandJoints2d(4, i) = fit_input[2 * (i + num_body_joint) + 1];
    }
    for (uint i = 0; i < num_hand_joint; i++)
    {
        RHandJoints2d(0, i) = 0;
        RHandJoints2d(1, i) = 0;
        RHandJoints2d(2, i) = 0;
        RHandJoints2d(3, i) = fit_input[2 * (i + num_hand_joint + num_body_joint)];
        RHandJoints2d(4, i) = fit_input[2 * (i + num_hand_joint + num_body_joint) + 1];
    }
    for (uint i = 0; i < num_face_landmark; i++)
    {
        Joints2d_face(0, i) = 0;
        Joints2d_face(1, i) = 0;
        Joints2d_face(2, i) = 0;
        Joints2d_face(3, i) = fit_input[2 * (i + num_hand_joint + num_body_joint + num_hand_joint)];
        Joints2d_face(4, i) = fit_input[2 * (i + num_hand_joint + num_body_joint + num_hand_joint) + 1];
    }
    const double* PAF_vec = fit_input + 2 * ModelFitter::NUM_KEYPOINTS_2D; 
    for (int i = 0; i < ModelFitter::NUM_PAF_VEC; i++)
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
    LFootJoints2d(3, 0) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0];  // Left BigToe
    LFootJoints2d(4, 0) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1];  // Left BigToe
    LFootJoints2d(3, 1) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2];  // Left SmallToe
    LFootJoints2d(4, 1) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3];  // Left SmallToe
    LFootJoints2d(3, 2) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4];  // Left Heel
    LFootJoints2d(4, 2) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5];  // Left Heel
    RFootJoints2d(3, 0) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 0 + 6];  // Right BigToe
    RFootJoints2d(4, 0) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 1 + 6];  // Right BigToe
    RFootJoints2d(3, 1) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 2 + 6];  // Right SmallToe
    RFootJoints2d(4, 1) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 3 + 6];  // Right SmallToe
    RFootJoints2d(3, 2) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 4 + 6];  // Right Heel
    RFootJoints2d(4, 2) = fit_input[2 * (num_hand_joint + num_body_joint + num_hand_joint + num_face_landmark) + 5 + 6];  // Right Heel

    Eigen::MatrixXd surface_constraint(6, 0);
    if (fit_surface)
    {
        const uint num_surface_constraint = densepose_result->size();
        surface_constraint.resize(6, num_surface_constraint);
        for (auto i = 0u; i < num_surface_constraint; i++)
        {
            std::fill(surface_constraint.data() + 6 * i + 0, surface_constraint.data() + 6 * i + 3, 0.0);
            surface_constraint.data()[6 * i + 3] = (*densepose_result)[i].x;
            surface_constraint.data()[6 * i + 4] = (*densepose_result)[i].y;
            surface_constraint.data()[6 * i + 5] = (*densepose_result)[i].z;
        }
    }

    Adam_Fit_PAF(adam, frame_params, bodyJoints2d, RFootJoints2d, LFootJoints2d, RHandJoints2d, LHandJoints2d, Joints2d_face, PAF, surface_constraint,
                 calibK, 2u, false, true, true, true);
}

void reconstruct_adam(TotalModel& adam, smpl::SMPLParams& frame_params, std::vector<double>& reconstruction, bool euler)
{
    const Eigen::VectorXd J_vec = adam.J_mu_ + adam.dJdc_ * frame_params.m_adam_coeffs;
    Eigen::VectorXd transforms_joint(3 * TotalModel::NUM_JOINTS * 4 + 3 * TotalModel::NUM_JOINTS);
    const double* p2t_parameters[2] = { frame_params.m_adam_pose.data(), J_vec.data() };
    double* p2t_residuals = transforms_joint.data();
    smpl::PoseToTransform_AdamFull_withDiff p2t(adam, std::array<std::vector<int>, TotalModel::NUM_JOINTS>(), euler); // the parent indexes can be arbitrary (not used)
    p2t.Evaluate(p2t_parameters, p2t_residuals, nullptr);

    // regressor_type == 2 !
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Vt(TotalModel::NUM_VERTICES, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outVerts(TotalModel::NUM_VERTICES, 3);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, 1> > Vt_vec(Vt.data(), 3 * TotalModel::NUM_VERTICES);
    Vt_vec = adam.m_meanshape + adam.m_shapespace_u * frame_params.m_adam_coeffs;
    adam_lbs(adam, Vt_vec.data(), transforms_joint, outVerts.data());
    outVerts.rowwise() += frame_params.m_adam_t.transpose();
    MatrixXdr J_coco = adam.m_small_coco_reg * outVerts;
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > outJoints(transforms_joint.data() + 4 * 3 * TotalModel::NUM_JOINTS, TotalModel::NUM_JOINTS, 3);
    outJoints.rowwise() += frame_params.m_adam_t.transpose();

    reconstruction.resize(3 * J_coco.rows() + 3 * 40 + 3 * 6);
    std::copy(J_coco.data(), J_coco.data() + 3 * J_coco.rows(), reconstruction.data());
    std::copy(p2t_residuals + 3 * TotalModel::NUM_JOINTS * 4 + 3 * 22, p2t_residuals + 3 * TotalModel::NUM_JOINTS * 4 + 3 * 62,
              reconstruction.data() + 3 * J_coco.rows()); // copy the fingers
    std::copy(outVerts.data() + 14328 * 3, outVerts.data() + 14328 * 3 + 3, reconstruction.data() + 3 * (J_coco.rows() + 40 + 0)); //right bigtoe
    std::copy(outVerts.data() + 14288 * 3, outVerts.data() + 14288 * 3 + 3, reconstruction.data() + 3 * (J_coco.rows() + 40 + 1)); //right littletoe
    reconstruction[3 * (J_coco.rows() + 40 + 2) + 0] = 0.5 * (outVerts.data()[3 * 14357 + 0] + outVerts.data()[3 * 14361 + 0]); // right heel
    reconstruction[3 * (J_coco.rows() + 40 + 2) + 1] = 0.5 * (outVerts.data()[3 * 14357 + 1] + outVerts.data()[3 * 14361 + 1]);
    reconstruction[3 * (J_coco.rows() + 40 + 2) + 2] = 0.5 * (outVerts.data()[3 * 14357 + 2] + outVerts.data()[3 * 14361 + 2]);
    std::copy(outVerts.data() + 12239 * 3, outVerts.data() + 12239 * 3 + 3, reconstruction.data() + 3 * (J_coco.rows() + 40 + 3)); //left bigtoe
    std::copy(outVerts.data() + 12289 * 3, outVerts.data() + 12289 * 3 + 3, reconstruction.data() + 3 * (J_coco.rows() + 40 + 4)); //left smalltoe
    reconstruction[3 * (J_coco.rows() + 40 + 5) + 0] = 0.5 * (outVerts.data()[3 * 12368 + 0] + outVerts.data()[3 * 12357 + 0]); // left heel
    reconstruction[3 * (J_coco.rows() + 40 + 5) + 1] = 0.5 * (outVerts.data()[3 * 12368 + 1] + outVerts.data()[3 * 12357 + 1]);
    reconstruction[3 * (J_coco.rows() + 40 + 5) + 2] = 0.5 * (outVerts.data()[3 * 12368 + 2] + outVerts.data()[3 * 12357 + 2]);
}

void refit_single_frame(TotalModel& adam, smpl::SMPLParams& frame_params, std::vector<double>& reconstruction, bool bFreezeShape, bool euler)
{
    assert(reconstruction.size() == 66 * 3);
    const uint num_body_joint = 21;
    const uint num_hand_joint = 21;
    const uint num_face_landmark = 70;
    Eigen::MatrixXd bodyJoints2d(5, num_body_joint);// (3, targetJoint.size());
    Eigen::MatrixXd Joints2d_face(5, num_face_landmark);// (3, landmarks_face.size());
    Eigen::MatrixXd LHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
    Eigen::MatrixXd RHandJoints2d(5, num_hand_joint);// (3, HandModel::NUM_JOINTS);
    Eigen::MatrixXd LFootJoints2d(5, 3);// (3, 3);      //Heel, Toe
    Eigen::MatrixXd RFootJoints2d(5, 3);// (3, 3);      //Heel, Toe
    Eigen::MatrixXd PAF(3, ModelFitter::NUM_PAF_VEC);
    PAF.setZero(); Joints2d_face.setZero();

    std::array<int, num_body_joint> cocoplus_to_smc = {{12, 14, 12, 9, 10, 11, 3, 4, 5, 8, 7, 6, 2, 1, 0, 15, 17, 16, 18, 13, 19}};
    for (uint i = 0; i < num_body_joint; i++)
    {
        bodyJoints2d(0, i) = reconstruction[3 * cocoplus_to_smc[i] + 0];
        bodyJoints2d(1, i) = reconstruction[3 * cocoplus_to_smc[i] + 1];
        bodyJoints2d(2, i) = reconstruction[3 * cocoplus_to_smc[i] + 2];
        bodyJoints2d(3, i) = 0;
        bodyJoints2d(4, i) = 0;
    }
    bodyJoints2d.col(2) = (bodyJoints2d.col(6) + bodyJoints2d.col(12)) / 2;  // central hip

    int offset = 20; // hand starts here in reconstruction
    LHandJoints2d(0, 0) = reconstruction[3 * 11 + 0];
    LHandJoints2d(1, 0) = reconstruction[3 * 11 + 1];
    LHandJoints2d(2, 0) = reconstruction[3 * 11 + 2];
    LHandJoints2d(3, 0) = 0;
    LHandJoints2d(4, 0) = 0;
    for (uint i = 1; i < num_hand_joint; i++)
    {
        LHandJoints2d(0, i) = reconstruction[3 * (i - 1 + offset) + 0];
        LHandJoints2d(1, i) = reconstruction[3 * (i - 1 + offset) + 1];
        LHandJoints2d(2, i) = reconstruction[3 * (i - 1 + offset) + 2];
        LHandJoints2d(3, i) = 0;
        LHandJoints2d(4, i) = 0;
    }

    offset = 20 + 20; // hand starts here in reconstruction
    RHandJoints2d(0, 0) = reconstruction[3 * 6 + 0];
    RHandJoints2d(1, 0) = reconstruction[3 * 6 + 1];
    RHandJoints2d(2, 0) = reconstruction[3 * 6 + 2];
    RHandJoints2d(3, 0) = 0;
    RHandJoints2d(4, 0) = 0;
    for (uint i = 1; i < num_hand_joint; i++)
    {
        RHandJoints2d(0, i) = reconstruction[3 * (i - 1 + offset) + 0];
        RHandJoints2d(1, i) = reconstruction[3 * (i - 1 + offset) + 1];
        RHandJoints2d(2, i) = reconstruction[3 * (i - 1 + offset) + 2];
        RHandJoints2d(3, i) = 0;
        RHandJoints2d(4, i) = 0;
    }

    LFootJoints2d.setZero(); RFootJoints2d.setZero();
    offset = 20 + 20 + 20;
    RFootJoints2d(0, 0) = reconstruction[3 * offset + 0];  // Left BigToe
    RFootJoints2d(1, 0) = reconstruction[3 * offset + 1];  // Left BigToe
    RFootJoints2d(2, 0) = reconstruction[3 * offset + 2];  // Left BigToe
    RFootJoints2d(0, 1) = reconstruction[3 * offset + 3];  // Left SmallToe
    RFootJoints2d(1, 1) = reconstruction[3 * offset + 4];  // Left SmallToe
    RFootJoints2d(2, 1) = reconstruction[3 * offset + 5];  // Left SmallToe
    RFootJoints2d(0, 2) = reconstruction[3 * offset + 6];  // Left Heel
    RFootJoints2d(1, 2) = reconstruction[3 * offset + 7];  // Left Heel
    RFootJoints2d(2, 2) = reconstruction[3 * offset + 8];  // Left Heel

    offset = 20 + 20 + 20 + 3;
    LFootJoints2d(0, 0) = reconstruction[3 * offset + 0];  // Left BigToe
    LFootJoints2d(1, 0) = reconstruction[3 * offset + 1];  // Left BigToe
    LFootJoints2d(2, 0) = reconstruction[3 * offset + 2];  // Left BigToe
    LFootJoints2d(0, 1) = reconstruction[3 * offset + 3];  // Left SmallToe
    LFootJoints2d(1, 1) = reconstruction[3 * offset + 4];  // Left SmallToe
    LFootJoints2d(2, 1) = reconstruction[3 * offset + 5];  // Left SmallToe
    LFootJoints2d(0, 2) = reconstruction[3 * offset + 6];  // Left Heel
    LFootJoints2d(1, 2) = reconstruction[3 * offset + 7];  // Left Heel
    LFootJoints2d(2, 2) = reconstruction[3 * offset + 8];  // Left Heel

    Adam_skeletal_refit(adam, frame_params, bodyJoints2d, RFootJoints2d, LFootJoints2d, RHandJoints2d, LHandJoints2d, Joints2d_face, PAF,
                        2, bFreezeShape, euler);
}
