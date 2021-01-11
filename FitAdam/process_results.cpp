#include <iostream>
#include <fstream>
#include <cstdio>
#include <map>
#include <gflags/gflags.h>
#include <vector>
#include <array>
#include <json/json.h>
#include <simple.h>
#include "totalmodel.h"
#include <FitToBody.h>
#include <VisualizedData.h>
#include <KinematicModel.h>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pose_to_transforms.h>
#include "meshTrackingProj.h"
#include "SGSmooth.hpp"
#include "ModelFitter.h"
#include "utils.h"
#include <thread>
#include <boost/filesystem.hpp>

DEFINE_string(root_dirs, "", "Base root folder to access data");
DEFINE_string(seqName, "default", "Sequence Name to run");
DEFINE_int32(start, 1, "Starting frame");
DEFINE_int32(end, 1000, "Ending frame");

TotalModel g_total_model;
const int NUM_JOINT_PARAMS = 21 * 3 + 2 * 21 * 3;
double gResultJoint[NUM_JOINT_PARAMS];

std::array<int, 19> map_totalcap_to_coco = {1, 0, 8, 5, 6, 7, 12, 13, 14, 2, 3, 4, 9, 10, 11, 16, 18, 15, 17}; // 19 body indices of BODY_25 from openpose
// feet indices are in order of openpose
std::array<int, 8> feet_vtx_idx = { 12239, //left bigtoe
                                    12289, //left littletoe
                                    12368, //left heel
                                    12357, //left heel
                                    14238, //right bigtoe
                                    14288, //right littletoe
                                    14357, //right heel
                                    14361 //right heel
                                  };
const int LEFT_HEEL_IDX = 21;
const int RIGHT_HEEL_IDX = 24;

std::map<int, std::string> POSE_BODY_25_BODY_PARTS = {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "MidHip"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    {25, "Background"}
};

std::map<int, std::string> POSE_SMPL_BODY_PARTS = {
    {0,  "hips"},
    {1,  "leftUpLeg"},
    {2,  "rightUpLeg"},
    {3,  "spine"},
    {4,  "leftLeg"},
    {5,  "rightLeg"},
    {6,  "spine1"},
    {7,  "leftFoot"},
    {8,  "rightFoot"},
    {9,  "spine2"},
    {10, "leftToeBase"},
    {11, "rightToeBase"},
    {12, "neck"},
    {13, "leftShoulder"},
    {14, "rightShoulder"},
    {15, "head"},
    {16, "leftArm"},
    {17, "rightArm"},
    {18, "leftForeArm"},
    {19, "rightForeArm"},
    {20, "leftHand"},
    {21, "rightHand"}
};


void check_flags(int argc, char* argv[])
{
#ifdef GFLAGS_NAMESPACE
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
#else
    google::ParseCommandLineFlags(&argc, &argv, true);
#endif
    std::cout << "Root Directory: " << FLAGS_root_dirs << std::endl;
    std::cout << "Sequence Name: " << FLAGS_seqName << std::endl;
    if (FLAGS_seqName.compare("default") == 0)
    {
        std::cerr << "Error: Sequence Name must be set." << std::endl;
        exit(1);
    }
    if (FLAGS_start >= FLAGS_end)
    {
        std::cerr << "Error: Starting frame must be less than end frame." << std::endl;
        exit(1);
    }
}

std::vector<smpl::SMPLParams> readResultFrames(std::string resDirName, int startFrame, int endFrame) {
    int numFrames = endFrame - startFrame;
    std::vector<smpl::SMPLParams> modelParams(numFrames);
    for (auto i = 0u, image_index = startFrame + i; i < numFrames; i++, image_index++) {
        std::cout << "Reading single frame results: " << image_index << std::endl;
        char basename[200];
        sprintf(basename, "%04d.txt", image_index);
        const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + resDirName + "/" + basename;
        smpl::SMPLParams frame_params;
        readFrameParam(param_filename, frame_params);
        modelParams[i] = frame_params;
    }

    return modelParams;
}

struct ResultFrame {
    cv::Point3d m_globalTrans; // 3 vec
    std::vector< cv::Point3d > m_jointsPos; // 25 x 3 b/c 25 joints in BODY_25
    std::vector< cv::Point3d > m_SMPLJointsPos; // 22 x 3 b/c 22 body joints in ADAM
    std::vector< cv::Point3d > m_SMPLJointsRot; // 22 x 3 b/c 22 body joints in ADAM
    std::vector< double > m_bodyCoeffs; // 30 coefficients describing body shape
    std::vector< double > m_faceCoeffs; // 200 coefficients describing face

    ResultFrame() 
        : m_globalTrans(), m_jointsPos(25), m_SMPLJointsPos(22), m_SMPLJointsRot(22), m_bodyCoeffs(TotalModel::NUM_SHAPE_COEFFICIENTS), m_faceCoeffs(TotalModel::NUM_EXP_BASIS_COEFFICIENTS)
        {}

    Json::Value serialize(int id) {
        Json::Value frame;
        frame["id"] = id;
        // global translation
        Json::Value transVal;
        transVal["x"] = m_globalTrans.x;
        transVal["y"] = m_globalTrans.y;
        transVal["z"] = m_globalTrans.z;
        frame["trans"] = transVal;

        // joint information
        for (int i = 0; i < m_jointsPos.size(); i++) {
            Json::Value curJointVal;
            curJointVal["name"] = POSE_BODY_25_BODY_PARTS[i];
            Json::Value jointPosVal;
            jointPosVal["x"] = m_jointsPos[i].x;
            jointPosVal["y"] = m_jointsPos[i].y;
            jointPosVal["z"] = m_jointsPos[i].z;
            curJointVal["pos"] = jointPosVal;

            frame["joints"].append(curJointVal);
        }

        // SMPL body info
        for (int i = 0; i < m_SMPLJointsPos.size(); i++) {
            Json::Value curJointVal;
            curJointVal["name"] = POSE_SMPL_BODY_PARTS[i];
            Json::Value jointPosVal;
            Json::Value jointRotVal;
            jointPosVal["x"] = m_SMPLJointsPos[i].x;
            jointPosVal["y"] = m_SMPLJointsPos[i].y;
            jointPosVal["z"] = m_SMPLJointsPos[i].z;
            curJointVal["pos"] = jointPosVal;

            jointRotVal["x"] = m_SMPLJointsRot[i].x;
            jointRotVal["y"] = m_SMPLJointsRot[i].y;
            jointRotVal["z"] = m_SMPLJointsRot[i].z;
            curJointVal["rot"] = jointRotVal;

            frame["SMPLJoints"].append(curJointVal);
        }

        // Shape coefficients
        for (int i = 0; i < m_bodyCoeffs.size(); i++) {
            frame["bodyCoeffs"].append(m_bodyCoeffs[i]);
        }
        for (int i = 0; i < m_faceCoeffs.size(); i++) {
            frame["faceCoeffs"].append(m_faceCoeffs[i]);
        }

        return frame;
    }
};

std::vector<ResultFrame> processResults(std::vector<smpl::SMPLParams> &modelParams) {
    int numFrames = modelParams.size();
    std::vector<ResultFrame> resultDataList(numFrames);
    CMeshModelInstance mesh;
    for (int i = 0; i < numFrames; i++) {
        mesh.clearMesh();
        GenerateMesh(mesh, gResultJoint, modelParams[i], g_total_model, 2, false, true); // use axis-angle and get local joint info
        
        // first openpose BODY_25 joints
        ResultFrame data;
        // root translation
        data.m_globalTrans = cv::Point3d(modelParams[i].m_adam_t[0], modelParams[i].m_adam_t[1], modelParams[i].m_adam_t[2]);
        // body joints are returned
        for (int j = 0; j < 19; j++) {
            data.m_jointsPos[map_totalcap_to_coco[j]] = cv::Point3d(gResultJoint[j*3], gResultJoint[j*3 + 1], gResultJoint[j*3 + 2]);
        }
        // feet joints are taken from vertices
        for (int j = 19, foot_idx = 0; j < 25; j++, foot_idx++) {
            cv::Point3d vtx1 = mesh.m_vertices[feet_vtx_idx[foot_idx]];
            if (j == RIGHT_HEEL_IDX || j == LEFT_HEEL_IDX) {
                // must average the vertices
                cv::Point3d vtx2 = mesh.m_vertices[feet_vtx_idx[++foot_idx]];
                data.m_jointsPos[j] = cv::Point3d(0.5 * (vtx1.x + vtx2.x), 0.5 * (vtx1.y + vtx2.y), 0.5 * (vtx1.z + vtx2.z));
            } else {
                data.m_jointsPos[j] = vtx1;
            }
        }

        // then SMPL joints
        for (int j = 0; j < 22; j++) {
            data.m_SMPLJointsPos[j] = mesh.m_joints[j];
            // if (j == 3 || j == 6 || j == 9) {
            //     std::cout << j << " regressed" << std::endl;
            //     std::cout << "(" << mesh.m_joints[j].x << ", " << mesh.m_joints[j].y << ", " << mesh.m_joints[j].z << ")\n";
            // }
        }
        for (int j = 0; j < 22; j++) {
            data.m_SMPLJointsRot[j] = cv::Point3d(modelParams[i].m_adam_pose(j, 0), modelParams[i].m_adam_pose(j, 1), modelParams[i].m_adam_pose(j, 2));
        }

        // shape coefficients
        for (int j = 0; j < TotalModel::NUM_SHAPE_COEFFICIENTS; j++) {
            data.m_bodyCoeffs[j] = modelParams[i].m_adam_coeffs(j, 0);
        }
        for (int j = 0; j < TotalModel::NUM_EXP_BASIS_COEFFICIENTS; j++) {
            data.m_faceCoeffs[j] = modelParams[i].m_adam_facecoeffs_exp(j, 0);
        }

        resultDataList[i] = data;
    }

    return resultDataList;
}

std::string serializeResults(std::vector<ResultFrame> &resultsList) {
    Json::Value root;
    for (int i = 0; i < resultsList.size(); i++) {
        root["totalcapResults"].append(resultsList[i].serialize(i));
    }

    Json::StyledWriter styledWriter;
    return styledWriter.write(root);
}

/**
Goes through a given directory of total capture output and processes the results to output BODY_25 (same as used by OpenPose) and SMPL joint data. 
 */
int main(int argc, char* argv[])
{
    check_flags(argc, argv);

    // initialize total model
    LoadTotalModelFromObj(g_total_model, std::string("model/mesh_nofeet.obj"));
    LoadModelColorFromObj(g_total_model, std::string("model/nofeetmesh_byTomas_bottom.obj"));  // contain the color information
    LoadTotalDataFromJson(g_total_model, std::string("model/adam_v1_plus2.json"), std::string("model/adam_blendshapes_348_delta_norm.json"), std::string("model/correspondences_nofeet.txt"));
    LoadCocoplusRegressor(g_total_model, std::string("model/regressor_0n1_root.json"));

    // read in fitting results for all frames (both before and after tracking)
    std::vector<smpl::SMPLParams> trackedModelParams = readResultFrames("body_3d_frontal_tracking", FLAGS_start, FLAGS_end);
    std::vector<smpl::SMPLParams> noTrackedModelParams = readResultFrames("body_3d_frontal", FLAGS_start, FLAGS_end);

    // go through each frame, build mesh, and collect info to output
    std::vector<ResultFrame> trackedResults = processResults(trackedModelParams);
    std::vector<ResultFrame> noTrackedResults = processResults(noTrackedModelParams);

    // write out
    std::string trackedResStr = serializeResults(trackedResults);
    std::string noTrackedResStr = serializeResults(noTrackedResults);

    const std::string tracked_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/tracked_results.json";
    std::ofstream fTrack(tracked_filename);
    if (!fTrack.good())
    {
        std::cerr << "Error: could not open json file for writing" << std::endl;
        exit(1);
    }
    fTrack << trackedResStr;
    fTrack.close();

    const std::string untracked_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/untracked_results.json";
    std::ofstream fNoTrack(untracked_filename);
    if (!fNoTrack.good())
    {
        std::cerr << "Error: could not open json file for writing" << std::endl;
        exit(1);
    }
    fNoTrack << noTrackedResStr;
    fNoTrack.close();
}
