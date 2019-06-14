#include "utils.h"
#include <iostream>
#include <cmath>
#include <vector>

void model_size(const double* const joint, const std::vector<int>& connMat)
{
	double lhand_size = 0, body_size = 0, rhand_size = 0;
	for (auto i = 0u; i < connMat.size(); i += 2)
	{
		const double length2 = (joint[3*connMat[i]] - joint[3*connMat[i+1]]) * (joint[3*connMat[i]] - joint[3*connMat[i+1]])
						+ (joint[3*connMat[i] + 1] - joint[3*connMat[i+1] + 1]) * (joint[3*connMat[i] + 1] - joint[3*connMat[i+1] + 1])
						+ (joint[3*connMat[i] + 2] - joint[3*connMat[i+1] + 2]) * (joint[3*connMat[i] + 2] - joint[3*connMat[i+1] + 2]);
		const double length = sqrt(length2);
		if ((i >= 4 && i < 8) || (i >= 10 && i < 14) || (i >= 18 && i < 22) || (i >= 24 && i < 28))
			body_size += length;
		else if (i >= 36 && i < 76)
			lhand_size += length;
		else if (i >= 76)
			rhand_size += length;
	}
	std::cout << "body size: " << body_size << " lhand size: " << lhand_size << " rhand size: " << rhand_size << std::endl;
}

cv::Mat alignMeshImage(const cv::Mat& meshImage, const cv::Mat& srcImage)
{
	assert(meshImage.cols == srcImage.cols && meshImage.rows == srcImage.rows);
	assert(meshImage.type() == CV_8UC3);
	assert(srcImage.type() == CV_8UC3 || srcImage.type() == CV_8UC1);
	cv::Mat mask_array[3];
	cv::Mat ret = srcImage.clone();
	cv::Mat bgmask, fgmask;
	cv::Mat foreGround = meshImage.clone();
	cv::compare(meshImage, cv::Scalar(255, 255, 255), bgmask, cv::CMP_EQ);  // background mask, 255, 3 channels
	cv::split(bgmask, mask_array);
	bgmask = mask_array[0];
	bgmask.mul(mask_array[1]);
	bgmask.mul(mask_array[2]);
	cv::bitwise_not(bgmask, fgmask);
	bgmask = bgmask / 255;
	fgmask = fgmask / 255;
	if (srcImage.type() == CV_8UC3)
	{
		mask_array[0] = mask_array[1] = mask_array[2] = bgmask;
		cv::merge(mask_array, 3, bgmask);
		mask_array[0] = mask_array[1] = mask_array[2] = fgmask;
		cv::merge(mask_array, 3, fgmask);
	}
	else
		cv::cvtColor(foreGround, foreGround, CV_BGR2GRAY);

	ret.convertTo(ret, CV_32F);
	foreGround.convertTo(foreGround, CV_32F);
	bgmask.convertTo(bgmask, CV_32F);
	fgmask.convertTo(fgmask, CV_32F);

	cv::multiply(ret, bgmask, ret);
	cv::multiply(foreGround, fgmask, foreGround);
	cv::add(ret, foreGround, ret);

	ret.convertTo(ret, srcImage.type());
	return ret;
}

cv::Mat alignMeshImage(const cv::Mat& meshImage, const cv::Mat& srcImage, const cv::Mat_<float> depthMap)
{
	assert(meshImage.cols == srcImage.cols && meshImage.rows == srcImage.rows);
	assert(meshImage.type() == CV_8UC3);
	assert(srcImage.type() == CV_8UC3 || srcImage.type() == CV_8UC1);
	assert(meshImage.cols == depthMap.cols && meshImage.rows == depthMap.rows);

	cv::Mat ret = srcImage.clone();
	cv::Mat bgmask, fgmask, foreGround;
	cv::compare(depthMap, cv::Scalar(999), fgmask, cv::CMP_LT);  // fg region is 255
	fgmask = fgmask / 255;
	bgmask = 1 - fgmask;

	if (srcImage.type() == CV_8UC3)
	{
		cv::Mat mask_array[3] = {bgmask, bgmask, bgmask};
		cv::merge(mask_array, 3, bgmask);
		mask_array[0] = mask_array[1] = mask_array[2] = fgmask;
		cv::merge(mask_array, 3, fgmask);
	}
	else
		cv::cvtColor(foreGround, foreGround, CV_BGR2GRAY);

	ret.convertTo(ret, CV_32F);
	meshImage.convertTo(foreGround, CV_32F);
	fgmask.convertTo(fgmask, CV_32F);
	bgmask.convertTo(bgmask, CV_32F);

	cv::multiply(ret, bgmask, ret);
	cv::multiply(foreGround, fgmask, foreGround);
	cv::add(ret, foreGround, ret);

	ret.convertTo(ret, srcImage.type());
	return ret;
}

cv::Mat alignMeshImageAlpha(const cv::Mat& meshImage, const cv::Mat& srcImage)
{
	// meshImage should be RGBA, srcImage should be RGB
	assert(meshImage.type() == CV_8UC4);
	assert(srcImage.type() == CV_8UC1 || srcImage.type() == CV_8UC3);

	cv::Mat mask_array[4];
	cv::Mat foreGround;
	cv::split(meshImage, mask_array);
	cv::Mat fgmask = mask_array[3];
	fgmask.convertTo(fgmask, CV_32F);
	fgmask = fgmask / 255.0f;
	// for (auto y = 0; y < fgmask.rows; y++)
	// 	for (auto x = 0; x < fgmask.cols; x++)
	// 	{
	// 		if (fgmask.at<float>(y, x) != 0.0)
	// 			fgmask.at<float>(y, x) = 1;
	// 	}
	cv::Mat bgmask = 1.0f - fgmask;
	cv::merge(mask_array, 3, foreGround);
	cv::cvtColor(foreGround, foreGround, CV_RGB2BGR);
	foreGround.convertTo(foreGround, CV_32F);
	cv::Mat ret = srcImage.clone();
	ret.convertTo(ret, CV_32F);

	if (srcImage.type() == CV_8UC3)
	{
		cv::Mat mask_array[3] = {bgmask, bgmask, bgmask};
		cv::merge(mask_array, 3, bgmask);
		mask_array[0] = mask_array[1] = mask_array[2] = fgmask;
		cv::merge(mask_array, 3, fgmask);
	}
	else
		cv::cvtColor(foreGround, foreGround, CV_BGR2GRAY);

	cv::multiply(ret, bgmask, ret);
	cv::multiply(foreGround, fgmask, foreGround);
	cv::add(ret, foreGround, ret);
	ret.convertTo(ret, srcImage.type());

	return ret;
}