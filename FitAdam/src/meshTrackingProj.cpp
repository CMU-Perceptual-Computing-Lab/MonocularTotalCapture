#include "meshTrackingProj.h"
#include <random>
#include <algorithm>
#include "opencv2/gpu/gpu.hpp"
// #define VISUALIZE_TRACKING

const float depthThresh = 1e-2;  // threshold for determining visibility
const float backward_check_thresh = 2.0;        //2.0 pixel
float RGBFLOAT_BUFFER[1920 * 1080 * 3];
// GLubyte RGB_BUFFER[1920 * 1080 * 3];

void Tracking_MeshVertex_depthMap(
const bool brox,
const cv::Mat& sourceImg,
const cv::Mat& targetImg,
const cv::Mat_<float>& depthMap,
const double* K,
const std::vector<cv::Point3d>& vertices,
std::vector<cv::Point3i>& target_constraints,
const uint sample_dist)  // If sample_dist > 0, only return 1 constraint in every (sample_dist, sample_dist) square, else return all valid constraint.
{
    if (sourceImg.type() != CV_8UC1 || targetImg.type() != CV_8UC1)
    {
        std::cerr << "Source image and target image must be grayscale" << std::endl;
        exit(0);
    }
    assert(K);

    std::vector<cv::Point2f> pt2d_source;
    std::vector<uint> vertices_idx;

    for (auto iv = 0u; iv < vertices.size(); iv++)
    {
        cv::Point2d pt2d;
        bool bvisible = ComputePtVisibilityUsingDepthMap(vertices[iv], K, depthMap, pt2d);
        if (!bvisible)
            continue;
        // visible vertices
        pt2d_source.push_back(cv::Point2f(pt2d.x, pt2d.y));
        vertices_idx.push_back(iv);
    }

    std::vector<cv::Point3i> new_constraints;
    if (brox)
    {
        // use Brox optical flow
        // compute optical from from virtual Image to the target Img
        cv::gpu::GpuMat frame0(sourceImg), frame1(targetImg);
        frame0.convertTo(frame0, CV_32F, 1.0 / 255.0);
        frame1.convertTo(frame1, CV_32F, 1.0 / 255.0);
        const double brox_alpha = 0.197;
        const double brox_gamma = 50.0;
        const double brox_scale = 0.8;
        const int brox_inner = 10;
        const int brox_outer = 70;
        const int brox_solver = 10;
        cv::gpu::BroxOpticalFlow brox_flow(brox_alpha, brox_gamma, brox_scale, brox_inner, brox_outer, brox_solver);

        cv::gpu::GpuMat cuda_fu, cuda_fv, cuda_bu, cuda_bv;
        brox_flow(frame0, frame1, cuda_fu, cuda_fv);
        brox_flow(frame1, frame0, cuda_bu, cuda_bv);
        cv::Mat fu(cuda_fu), fv(cuda_fv), bu(cuda_bu), bv(cuda_bv);

        cv::Mat cover;
        if (sample_dist > 0)
        {
            cover.create(sourceImg.rows / sample_dist + 1, sourceImg.cols / sample_dist + 1, CV_8UC1);
            cover.setTo(cv::Scalar(0));
        }
        for (uint iv2 = 0; iv2 < pt2d_source.size(); iv2++)
        {
            const int x = round(pt2d_source[iv2].x);
            const int y = round(pt2d_source[iv2].y);
            const int dest_x = round(x + fu.at<float>(y, x));
            const int dest_y = round(y + fv.at<float>(y, x));
            if (dest_x < 0 || dest_x >= 1920 || dest_y < 0 || dest_y >= 1080)
                continue;
            const int back_x = dest_x + round(bu.at<float>(dest_y, dest_x));
            const int back_y = dest_y + round(bv.at<float>(dest_y, dest_x));
            const float dist = sqrt((x - back_x) * (x - back_x) + (y - back_y) * (y - back_y));
            if (dist > backward_check_thresh)
                continue;
            if (sample_dist > 0)
            {
                if (cover.at<uchar>(int(dest_y / sample_dist), int(dest_x / sample_dist)))
                    continue;
                cover.at<uchar>(int(dest_y / sample_dist), int(dest_x / sample_dist)) = 1;
            }
            new_constraints.push_back(cv::Point3i(dest_x, dest_y, vertices_idx[iv2]));
        }
#ifdef VISUALIZE_TRACKING
    cv::Mat warped(sourceImg.rows, sourceImg.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < sourceImg.rows; y++)
        for (int x = 0; x < sourceImg.cols; x++)
        {
            const int dest_x = round(x + fu.at<float>(y, x));
            const int dest_y = round(y + fv.at<float>(y, x));
            if (dest_x < 0 || dest_x >= 1920 || dest_y < 0 || dest_y >= 1080)
                continue;
            const int back_x = dest_x + round(bu.at<float>(dest_y, dest_x));
            const int back_y = dest_y + round(bv.at<float>(dest_y, dest_x));
            const float dist = sqrt((x - back_x) * (x - back_x) + (y - back_y) * (y - back_y));
            if (dist > backward_check_thresh)
                continue;
            warped.at<uchar>(dest_y, dest_x) = sourceImg.at<uchar>(y, x);
        }
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
        cv::imshow( "Display window", warped );
        cv::waitKey(0);
#endif
    }
    else
    {
        std::vector<cv::Point2f> pt2d_target;
        std::vector<cv::Point2f> pt2d_backtracked;
        std::vector<uchar> outputStatus;
        std::vector<float> outputError;
        std::vector<uchar> outputStatus_2;
        std::vector<float> outputError_2;

        pt2d_target.reserve(pt2d_source.size());
        pt2d_backtracked.reserve(pt2d_source.size());
        outputStatus.reserve(pt2d_source.size());
        outputError.reserve(pt2d_source.size());
        outputStatus_2.reserve(pt2d_source.size());
        outputError_2.reserve(pt2d_source.size());

        // use LK optical flow
        const int pyrSize = 25;
        //forward tracking
        cv::calcOpticalFlowPyrLK(sourceImg, targetImg,
            pt2d_source, pt2d_target,
            outputStatus, outputError, cvSize(pyrSize, pyrSize), 3);

        //backward checking
        cv::calcOpticalFlowPyrLK(targetImg, sourceImg,
            pt2d_target, pt2d_backtracked,
            outputStatus_2, outputError_2, cvSize(pyrSize, pyrSize), 3);

        cv::Mat cover;
        if (sample_dist > 0)
        {
            cover.create(sourceImg.rows / sample_dist + 1, sourceImg.cols / sample_dist + 1, CV_8UC1);
            cover.setTo(cv::Scalar(0));
        }
        for (uint iv2 = 0; iv2 < pt2d_source.size(); iv2++)
        {
            if (!outputStatus[iv2]) continue;
            const float backtrackDist = sqrt( (pt2d_backtracked[iv2].x - pt2d_source[iv2].x) * (pt2d_backtracked[iv2].x - pt2d_source[iv2].x) +
                                              (pt2d_backtracked[iv2].y - pt2d_source[iv2].y) * (pt2d_backtracked[iv2].y - pt2d_source[iv2].y) );
            if (backtrackDist > backward_check_thresh)
                outputStatus[iv2] = 0;
            else
            {
                if (sample_dist > 0)
                {
                    if (!cover.at<uchar>(int(pt2d_target[iv2].y / sample_dist), int(pt2d_target[iv2].x / sample_dist)))
                    {
                        new_constraints.push_back(cv::Point3i(pt2d_target[iv2].x, pt2d_target[iv2].y, vertices_idx[iv2]));
                        cover.at<uchar>(int(pt2d_target[iv2].y / sample_dist), int(pt2d_target[iv2].x / sample_dist)) = 1;
                    }
                }
                else
                    new_constraints.push_back(cv::Point3i(pt2d_target[iv2].x, pt2d_target[iv2].y, vertices_idx[iv2]));
            }
        }
    }

    // copy data 
    target_constraints.insert(target_constraints.end(), new_constraints.begin(), new_constraints.end());

#ifdef VISUALIZE_TRACKING
    std::cout << "Number of constraints: " << new_constraints.size() << std::endl;
    cv::Mat vis;
    cv::hconcat(sourceImg, targetImg, vis);
    for (uint iv3 = 0; iv3 < new_constraints.size(); iv3++)
    {
        cv::line(vis, ProjectPt(vertices[new_constraints[iv3].z], K), cv::Point2f(new_constraints[iv3].x + targetImg.cols, new_constraints[iv3].y), 255);
    }
    // cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "Display window", depthMap / 1000 );
    // cv::waitKey(0);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", vis );
    cv::waitKey(0);
#endif
}

bool ComputePtVisibilityUsingDepthMap(
const cv::Point3d pt3d,
const double* K,
const cv::Mat_<float>& depthMap,
cv::Point2d& pt2d)
{
    pt2d = ProjectPt(pt3d, K);
    const int newX = round(pt2d.x), newY = round(pt2d.y);
    if (newX < 0 || newY < 0 || newX >= depthMap.cols || newY >= depthMap.rows)
        return false;  // out of image boundary
    const float depth = depthMap.at<float>(newY, newX);
    if ((pt3d.z - depth) / depth> depthThresh)
        return false;
    else
        return true;
}

cv::Point2d ProjectPt(cv::Point3d pt, const double* K)
{
    // Project a 3D point using currect projection matrix K
    const double x = pt.x * K[0] + pt.y * K[1] + pt.z * K[2];
    const double y = pt.x * K[3] + pt.y * K[4] + pt.z * K[5];
    const double z = pt.x * K[6] + pt.y * K[7] + pt.z * K[8];
    return cv::Point2d(x / z, y / z);
}

void createVirtualImage(
std::unique_ptr<Renderer>& render,
const double* K,
const CMeshModelInstance& mesh1,
const CMeshModelInstance& mesh2,
const cv::Mat_<float>& depthMap1,   // depthmap of mesh1
const cv::Mat& sourceImg,
cv::Mat& resultImg,
cv::Mat& XY,
const bool background)   // whether to use background from the source image
{
    assert(render);
    assert(mesh1.m_vertices.size() == mesh2.m_vertices.size());

    VisualizedData vis_data;
    vis_data.read_rgbfloat_buffer = RGBFLOAT_BUFFER;
    CopyMesh(mesh2, vis_data);

    for (auto iv = 0u; iv < mesh1.m_vertices.size(); iv++)
    {
        cv::Point2d target;
        const bool bvisibility = ComputePtVisibilityUsingDepthMap(mesh1.m_vertices[iv], K, depthMap1, target);
        if (bvisibility)
            vis_data.m_meshVerticesColor[iv] = cv::Point3d(target.x, target.y, 0.);   // change the color from RGB to our desired value
        else
            vis_data.m_meshVerticesColor[iv] = cv::Point3d(-10000., -10000., 0.);
    }

    render->CameraMode(0);
    double calibK[9];
    std::copy(K, K + 9, calibK);
    render->options.K = calibK;
    render->RenderProjection(vis_data);
    render->RenderAndReadProjection();
    XY = cv::Mat(sourceImg.rows, sourceImg.cols, CV_32FC3, RGBFLOAT_BUFFER);   // wrapper around buffer
    cv::flip(XY, XY, 0);

    cv::Mat XYc[3];
    cv::split(XY, XYc);

    resultImg.create(sourceImg.rows, sourceImg.cols, sourceImg.type());
    if (sourceImg.type() == CV_8UC1)
    {
        resultImg.setTo(cv::Scalar(0));
        for (auto y = 0; y < 1080; y++)
            for (auto x = 0; x < 1920; x++)
            {
                const int x0 = XYc[0].at<float>(y, x);
                const int y0 = XYc[1].at<float>(y, x);
                if (x0 >= 0 && x0 < 1920 && y0 >= 0 && y0 < 1080)
                {
                    resultImg.at<uchar>(y, x) = sourceImg.at<uchar>(y0, x0);
                }
                else if (background)
                    resultImg.at<uchar>(y, x) = sourceImg.at<uchar>(y, x);
                else
                    resultImg.at<uchar>(y, x) = 128;
            }
    }
    else if (sourceImg.type() == CV_8UC3)
    {
        // 
        resultImg.setTo(cv::Scalar(0));
        for (auto y = 0; y < 1080; y++)
            for (auto x = 0; x < 1920; x++)
            {
                const int x0 = XYc[0].at<float>(y, x);
                const int y0 = XYc[1].at<float>(y, x);
                if (x0 >= 0 && x0 < 1920 && y0 >= 0 && y0 < 1080)
                {
                    resultImg.at<cv::Vec3b>(y, x) = sourceImg.at<cv::Vec3b>(y0, x0);
                }
                else if (background)
                    resultImg.at<cv::Vec3b>(y, x) = sourceImg.at<cv::Vec3b>(y, x);
                else
                    resultImg.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
            }
    }
    else
    {
        std::cerr << "Unknown type of source image data type: " << sourceImg.type() << std::endl;
        exit(1);
    }
}

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
const uint sample_dist)
{
    assert(render);
    assert(mesh1.m_vertices.size() == mesh2.m_vertices.size());

    VisualizedData vis_data;
    vis_data.read_rgbfloat_buffer = RGBFLOAT_BUFFER;
    CopyMesh(mesh2, vis_data);

    for (auto iv = 0u; iv < mesh1.m_vertices.size(); iv++)
    {
        cv::Point2d target;
        const bool bvisibility = ComputePtVisibilityUsingDepthMap(mesh1.m_vertices[iv], K, depthMap1, target);
        if (bvisibility)
            vis_data.m_meshVerticesColor[iv] = cv::Point3d(target.x, target.y, 0.);   // change the color from RGB to our desired value
        else
            vis_data.m_meshVerticesColor[iv] = cv::Point3d(-10000., -10000., 0.);
    }

    render->CameraMode(0);
    double calibK[9];
    std::copy(K, K + 9, calibK);
    render->options.K = calibK;
    render->RenderProjection(vis_data);
    render->RenderAndReadProjection();
    cv::Mat XY = cv::Mat(sourceImg.rows, sourceImg.cols, CV_32FC3, RGBFLOAT_BUFFER);   // wrapper around buffer
    cv::flip(XY, XY, 0);

    cv::Mat XYc[3];
    cv::split(XY, XYc);

    resultImg.create(sourceImg.rows, sourceImg.cols, sourceImg.type());
    resultImg.setTo(cv::Scalar(0, 0, 0));
    int count_pixel = 0;
    for (auto y = 0; y < 1080; y++)
        for (auto x = 0; x < 1920; x++)
        {
            const int x0 = XYc[0].at<float>(y, x);
            const int y0 = XYc[1].at<float>(y, x);
            if (x0 >= 0 && x0 < 1920 && y0 >= 0 && y0 < 1080)
            {
                resultImg.at<uchar>(y, x) = sourceImg.at<uchar>(y0, x0);
                count_pixel++;
            }
            else
            {
                resultImg.at<uchar>(y, x) = targetImg.at<uchar>(y, x);
                // resultImg.at<uchar>(y, x) = sourceImg.at<uchar>(y, x);
            }
        }
    std::cout << "Copy from texture pixels: " << count_pixel << std::endl;

#ifdef VISUALIZE_TRACKING
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", depthMap2 / 999 );
    cv::waitKey(0);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", XYc[0] / 1920 );
    cv::waitKey(0);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", resultImg );
    cv::waitKey(0);
#endif

    // compute optical from from virtual Image to the target Img
    cv::gpu::GpuMat frame0(resultImg), frame1(targetImg);
    frame0.convertTo(frame0, CV_32F, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32F, 1.0 / 255.0);
    const double brox_alpha = 0.197;
    const double brox_gamma = 50.0;
    const double brox_scale = 0.8;
    const int brox_inner = 10;
    const int brox_outer = 70;
    const int brox_solver = 10;
    cv::gpu::BroxOpticalFlow brox_flow(brox_alpha, brox_gamma, brox_scale, brox_inner, brox_outer, brox_solver);

    cv::gpu::GpuMat cuda_fu, cuda_fv, cuda_bu, cuda_bv;
    brox_flow(frame0, frame1, cuda_fu, cuda_fv);
    brox_flow(frame1, frame0, cuda_bu, cuda_bv);

    cv::Mat fu(cuda_fu), fv(cuda_fv), bu(cuda_bu), bv(cuda_bv);
    cv::Mat warped(1080, 1920, CV_8UC1); warped.setTo(cv::Scalar(0));
    cv::Mat tracking_valid(1080, 1920, CV_8UC1, cv::Scalar(0));
    uint count_threshold = 0;
    for (auto y = 0; y < 1080; y++)
        for (auto x = 0; x < 1920; x++)
        {
            const int dest_x = x + round(fu.at<float>(y, x));
            const int dest_y = y + round(fv.at<float>(y, x));
            if (dest_x < 0 || dest_x >= 1920 || dest_y < 0 || dest_y >= 1080)
                continue;
            const int back_x = dest_x + round(bu.at<float>(dest_y, dest_x));
            const int back_y = dest_y + round(bv.at<float>(dest_y, dest_x));
            const float dist = sqrt((x - back_x) * (x - back_x) + (y - back_y) * (y - back_y));
            if (dist > backward_check_thresh)
            {
                count_threshold++;
                continue;
            }
            const int x0 = XYc[0].at<float>(y, x);
            const int y0 = XYc[1].at<float>(y, x);
            // warped.at<uchar>(dest_y, dest_x) = resultImg.at<uchar>(y, x);  // with the if statement, show only the pixel from the mesh.
            if (x0 >= 0 && x0 < 1920 && y0 >= 0 && y0 < 1080)
                warped.at<uchar>(dest_y, dest_x) = resultImg.at<uchar>(y, x);
            tracking_valid.at<uchar>(y, x) = 1;
        }
    std::cout << "Back tracking threshold exceeded: " << count_threshold << " points." << std::endl;
#ifdef VISUALIZE_TRACKING
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", targetImg );
    cv::waitKey(0);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", warped );
    cv::waitKey(0);
#endif

    cv::Mat cover;
    if (sample_dist > 0)
    {
        cover.create(sourceImg.rows / sample_dist + 1, sourceImg.cols / sample_dist + 1, CV_8UC1);
        cover.setTo(cv::Scalar(0));
    }
    int count_invisible = 0;
    for (auto iv = 0u; iv < mesh2.m_vertices.size(); iv++)
    {
        cv::Point2d pt2d;
        const bool bvisibility = ComputePtVisibilityUsingDepthMap(mesh2.m_vertices[iv], K, depthMap2, pt2d);
        if (!bvisibility) {count_invisible++; continue;}
        const int x = round(pt2d.x);
        const int y = round(pt2d.y);
        if (!tracking_valid.at<uchar>(y, x)) continue;
        if (sample_dist > 0)
            if (cover.at<uchar>(int(y / sample_dist), int(x / sample_dist)))
                continue;
        const int dest_x = x + round(fu.at<float>(y, x));
        const int dest_y = y + round(fv.at<float>(y, x));
        if (sample_dist > 0)
            cover.at<uchar>(int(y / sample_dist), int(x / sample_dist)) = 1;
        target_constraints.push_back(cv::Point3i(dest_x, dest_y, iv));
    }
    std::cout << "Invisible vertices: " << count_invisible << std::endl;
}

void downSampleConstraints(std::vector<cv::Point3i>& surface_constraint, const int sample_dist, const int height, const int width)
{
    if (sample_dist == 1)
        return;  // integer, cannot go dense than 1 in 1 x 1 square
    cv::Mat cover(height, width, CV_8UC1, cv::Scalar(0));
    for (auto iv = 0u; iv < surface_constraint.size(); iv++)
    {
        const int x = surface_constraint[iv].x;
        const int y = surface_constraint[iv].y;
        if (!cover.at<uchar>(int(y / sample_dist), int(x / sample_dist)))
            cover.at<uchar>(int(y / sample_dist), int(x / sample_dist)) = 1;
        else
        {
            surface_constraint.erase(surface_constraint.begin() + iv);
            iv--;
        }
    }
}
