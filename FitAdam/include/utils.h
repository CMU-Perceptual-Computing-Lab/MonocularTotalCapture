#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

void model_size(const double* const joint, const std::vector<int>& connMat);
cv::Mat alignMeshImage(const cv::Mat& meshImage, const cv::Mat& srcImage);
cv::Mat alignMeshImage(const cv::Mat& meshImage, const cv::Mat& srcImage, const cv::Mat_<float> depthMap);
cv::Mat alignMeshImageAlpha(const cv::Mat& meshImage, const cv::Mat& srcImage);
