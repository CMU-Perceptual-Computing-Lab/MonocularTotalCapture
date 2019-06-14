#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <Renderer.h>
#include <VisualizedData.h>
#include <CMeshModelInstance.h>
#include <KinematicModel.h>

void Tracking_MeshVertex_depthMap(
const bool brox,
const cv::Mat& sourceImg,
const cv::Mat& targetImg,
const cv::Mat_<float>& depthMap,
const double* K,
const std::vector<cv::Point3d>& vertices,
std::vector<cv::Point3i>& target_constraints,
const uint sample_dist=5u);

bool ComputePtVisibilityUsingDepthMap(
const cv::Point3d pt3d,
const double* K,
const cv::Mat_<float>& depthMap,
cv::Point2d& pt2d);

cv::Point2d ProjectPt(cv::Point3d pt, const double* K);

void createVirtualImage(
std::unique_ptr<Renderer>& render,
const double* K,
const CMeshModelInstance& mesh1,
const CMeshModelInstance& mesh2,
const cv::Mat_<float>& depthMap1,   // depthmap of mesh1
const cv::Mat& sourceImg,
cv::Mat& resultImg,
cv::Mat& XY,
const bool background=true);

void getVirtualImageConstraint(
std::unique_ptr<Renderer>& render,
const double* K,
const CMeshModelInstance& mesh1,
const CMeshModelInstance& mesh2,
const cv::Mat_<float>& depthMap1,   // depthmap of mesh1
const cv::Mat_<float>& depthMap2,   // depthmap of mesh2
const cv::Mat& sourceImg,
const cv::Mat& targetImg,
cv::Mat& resultImg,
std::vector<cv::Point3i>& target_constraints,
const uint sample_dist=1u);

void downSampleConstraints(std::vector<cv::Point3i>& surface_constraint, const int sample_dist=1, const int height=1080, const int width=1920);
