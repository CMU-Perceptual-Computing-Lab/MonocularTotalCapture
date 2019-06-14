#pragma once
#include <AdamFastCost.h>
#include <simple.h>
#include <FitCost.h>
#include <memory>
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <ceres/normal_prior.h>
#include <opencv2/opencv.hpp>
#include <DCTCost.h>

class ModelFitter
{
public:
    // 21 + 21 + 21 + 70 + 6
    const static int NUM_KEYPOINTS_2D = 139;
    // 14 + 3 + 6 + 20 + 20
    const static int NUM_PAF_VEC = 63;
    ModelFitter(const TotalModel& g_total_model);
    void setCalibK(const double* K);
    void initParameters(const smpl::SMPLParams& frame_params, const uint t=0);
    void readOutParameters(smpl::SMPLParams& frame_params, const uint t=0);
    void resetTimeStep(const uint t);
    void resetFitData();
    void resetCostFunction();
    void runFitting();
    // helper function
    void setSurfaceConstraints2D(const std::vector<cv::Point3i>& surfaceConstraint2D, const uint t=0);
    void setFitDataNetOutput(
        const std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2> net_output,
        const bool copyBody=true,
        const bool copyFoot=true,
        const bool copyHand=true,
        const bool copyFace=true,
        const bool copyPAF=true,
        const uint t=0
    );
    void setFitDataReconstruction(const std::vector<double>& reconstruction, const uint t=0);
    void multiStageFitting();

    std::vector<std::shared_ptr<AdamFullCost>> pCostFunction;
    AdamBodyPoseParamPriorDiff poseRegularizer;  // pose regularizer is here to facilitate adjusting weight
    ceres::Solver::Options options;

    // some fixed options for AdamFullCost
    int regressor_type;
    bool fit_face_exp;
    bool euler;
    // some fixed options for AdamFitData
    bool fit3D;
    bool fit2D;
    bool fitPAF;
    bool fit_surface;
    bool freezeShape;
    bool shareCoeff;
    // some options about smoothing
    uint DCT_trans_start;
    uint DCT_trans_end;
    uint DCT_trans_low_comp;
    // fit input data
    std::vector<Eigen::MatrixXd> bodyJoints;  // 21 joints SMC order
    std::vector<Eigen::MatrixXd> rFoot;  // large toe, small toe, 
    std::vector<Eigen::MatrixXd> lFoot;
    std::vector<Eigen::MatrixXd> faceJoints;   // 70 joints
    std::vector<Eigen::MatrixXd> lHandJoints;  // 21 joints
    std::vector<Eigen::MatrixXd> rHandJoints;
    std::vector<Eigen::MatrixXd> PAF;   // 63 PAFs
    std::vector<Eigen::MatrixXd> surface_constraint;
    // weight for regularizers / prior
    float wPoseRg, wCoeffRg, wPosePr, wHandPosePr, wFacePr;
private:
    void checkTimeStepEqual();
    const TotalModel& adam;
    std::vector<AdamFitData> FitData;
    std::array<double, 9> calibK;  // camera intrinsics
    bool costFunctionInit;
    bool calibKInit;
    bool fitDataInit;
    // parameter buffer for optimization
    std::vector<Eigen::Matrix<double, 62, 3, Eigen::RowMajor>> pose;
    std::vector<Eigen::Matrix<double, 30, 1>> coeffs;
    std::vector<Eigen::Vector3d> trans;
    std::vector<Eigen::Matrix<double, 200, 1>> exp;
    // ceres summary variable
    ceres::Solver::Summary summary;
    // Data for the prior
    Eigen::Matrix<double, 72, TotalModel::NUM_POSE_PARAMETERS> prior_A;
    Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> prior_mu;
    Eigen::MatrixXd hand_prior_A;   // have to use MatrixXd here, size too large too put on stack
    Eigen::Matrix<double, TotalModel::NUM_POSE_PARAMETERS, 1> hand_prior_mu;
    // regularizer / prior cost
    CoeffsParameterNormDiff coeffRegularizer;
    std::unique_ptr<ceres::NormalPrior> pPosePriorCost;
    std::unique_ptr<ceres::NormalPrior> pHandPosePriorCost;
    ceres::NormalPrior facePriorCost;
    uint timeStep;
    // smooth cost
    DCTCost* smooth_cost_trans;
    static const uint num_body_joint = 21;
    static const uint num_hand_joint = 21;
    static const uint num_face_landmark = 70;
};

// some separate functions
void fit_single_frame(TotalModel& adam, double* fit_input, double calibK[], smpl::SMPLParams& frame_params, std::vector<std::vector<cv::Point3i>>::iterator densepose_result, bool fit_surface);
void reconstruct_adam(TotalModel& adam, smpl::SMPLParams& frame_params, std::vector<double>& reconstruction, bool euler=true);
void refit_single_frame(TotalModel& adam, smpl::SMPLParams& frame_params, std::vector<double>& reconstruction, bool bFreezeShape, bool euler);