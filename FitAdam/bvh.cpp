#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <gflags/gflags.h>
#include <vector>
#include <array>
#include <json/json.h>
#include <simple.h>
#include "totalmodel.h"
#include <FitToBody.h>
#include <VisualizedData.h>
#include <Renderer.h>
#include <KinematicModel.h>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include <GL/freeglut.h>
#include <pose_to_transforms.h>
#include "meshTrackingProj.h"
#include "SGSmooth.hpp"
#include "ModelFitter.h"
#include "utils.h"
#include <thread>
#include <boost/filesystem.hpp>
#include <BVHWriter.h>

#define ROWS 1080
#define COLS 1920
#define FACE_VERIFY_THRESH 0.05
#define PI 3.14159265359

TotalModel g_total_model;

int main()
{
	// initialize total model
    LoadTotalModelFromObj(g_total_model, std::string("model/mesh_nofeet.obj"));
    LoadModelColorFromObj(g_total_model, std::string("model/nofeetmesh_byTomas_bottom.obj"));  // contain the color information
    LoadTotalDataFromJson(g_total_model, std::string("model/adam_v1_plus2.json"), std::string("model/adam_blendshapes_348_delta_norm.json"), std::string("model/correspondences_nofeet.txt"));
    LoadCocoplusRegressor(g_total_model, std::string("model/regressor_0n1_root.json"));

    // use the skeleton of the first frame
	const std::string param_filename = "../data/example_dance/body_3d_frontal/0001.txt";
	smpl::SMPLParams frame_params;
	readFrameParam(param_filename, frame_params);
	Eigen::VectorXd J_vec = g_total_model.J_mu_ + g_total_model.dJdc_ * frame_params.m_adam_coeffs;
	const Eigen::Matrix<double, 3 * TotalModel::NUM_JOINTS, 1> J0 = J_vec;

	// assume we already have the estimation result of the first 10 frames of the dancing sequence
	std::vector<Eigen::Matrix<double, 3, 1>> t;
	std::vector<Eigen::Matrix<double, TotalModel::NUM_JOINTS, 3, Eigen::RowMajor>> pose;
	for (auto i = 1; i <= 10; i++)
	{
		char frame_param_name[200];
		sprintf(frame_param_name, "../data/example_dance/body_3d_frontal/%04d.txt", i);
		readFrameParam(frame_param_name, frame_params);
		t.push_back(frame_params.m_adam_t);
		pose.push_back(frame_params.m_adam_pose);
	}

	BVHWriter bvh(g_total_model.m_parent);
	bvh.parseInput(J0, t, pose);
	bvh.writeBVH("output.bvh", 1.0 / 30);

	return 0;
}