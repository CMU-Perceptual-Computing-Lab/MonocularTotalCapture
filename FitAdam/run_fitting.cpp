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

#define ROWS 1080
#define COLS 1920
#define FACE_VERIFY_THRESH 0.05
#define PI 3.14159265359
DEFINE_string(root_dirs, "", "Base root folder to access data");
DEFINE_string(seqName, "default", "Sequence Name to run");
DEFINE_int32(start, 1, "Starting frame");
DEFINE_int32(end, 1000, "Ending frame");
DEFINE_bool(densepose, false, "Whether to fit onto result of densepose");
DEFINE_bool(OpenGLactive, false, "Whether to Stay in OpenGLWindow");
DEFINE_bool(euler, false, "True to use Euler angles, false to use angle axis representation");
DEFINE_int32(stage, 1, "Start from which stage.");
DEFINE_bool(imageOF, false, "If true, use image optical flow for the first tracking iteration; if false, always use texture optical flow.");
DEFINE_int32(freeze, 0, "If 1, do not use optical flow below hips; if 2, do not use optical for below chest.");
DEFINE_bool(singleStage, false, "If true, use single stage model fitter.");

TotalModel g_total_model;
double gResultJoint[21 * 3 + 2 * 21 * 3];
std::unique_ptr<Renderer> render = nullptr;
GLubyte ret_bytes[COLS * ROWS * 4];
float ret_depth[COLS * ROWS];

void emptyfunc() {}

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

void filter_hand_pose(const std::vector<smpl::SMPLParams>& params, std::vector<smpl::SMPLParams*>& batch_refit_params_ptr, uint start_frame)
{
    // run Savitzky-Golay filter on wrist and finger joints of params and copy to batch_refit_params_ptr
    assert(start_frame + batch_refit_params_ptr.size() <= params.size());
    if (batch_refit_params_ptr.size() < 2 * 11 + 2)
    {
        for (uint d = 60; d < TotalModel::NUM_POSE_PARAMETERS; d++)
        {
            for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
                batch_refit_params_ptr[t]->m_adam_pose.data()[d] = params[start_frame + t].m_adam_pose.data()[d];
        }
        return;
    }
    for (uint d = 60; d < TotalModel::NUM_POSE_PARAMETERS; d++)
    {
        const int order = (d < 66) ? 3 : 5;
        std::vector<double> input(batch_refit_params_ptr.size());
        // Eigen::VectorXf input(batch_refit_params_ptr.size());
        for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
            input.data()[t] = params[start_frame + t].m_adam_pose.data()[d];
        // Eigen::RowVectorXf output = savgolfilt(input, order, 21); // make sure the frame number is odd
        auto output = sg_smooth(input, 11, order);
        for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
            batch_refit_params_ptr[t]->m_adam_pose.data()[d] = output.data()[t];
    }
}

int main(int argc, char* argv[])
{
    check_flags(argc, argv);
    render.reset(new Renderer(&argc, argv));  // initialize the OpenGL renderer
    render->options.meshSolid = true;
    render->options.show_joint = false;
    Renderer::use_color_fbo = true;

    /*
    Stage 0: read in data
    */
    double calibK[9];  // K Matrix
    const std::string calib_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/calib.json";
    Json::Value json_root;
    std::ifstream f(calib_filename.c_str());
    if (!f.good())
    {
        std::cerr << "Error: Calib file " << calib_filename << " does not exists" << std::endl;
        exit(1);
    }
    f >> json_root;
    calibK[0] = json_root["K"][0u][0u].asDouble(); calibK[1] = json_root["K"][0u][1u].asDouble(); calibK[2] = json_root["K"][0u][2u].asDouble(); calibK[3] = json_root["K"][1u][0u].asDouble(); calibK[4] = json_root["K"][1u][1u].asDouble(); calibK[5] = json_root["K"][1u][2u].asDouble(); calibK[6] = json_root["K"][2u][0u].asDouble(); calibK[7] = json_root["K"][2u][1u].asDouble(); calibK[8] = json_root["K"][2u][2u].asDouble();
    f.close();

    std::vector<std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2>> net_output;   // read in network output
    for (int i = FLAGS_start; i < FLAGS_end; i++)
    {
        char basename[20];
        sprintf(basename, "%012d.txt", i);
        const std::string filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/net_output/" + basename;
        std::ifstream f2(filename.c_str());
        // check the file exists
        if (!f2.good())
        {
            std::cerr << "Error: File " << filename << " does not exists" << std::endl;
            exit(1);
        }

        std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2> net_output_entry;
        std::string str;
        std::getline(f2, str);
        if (str.compare("2D keypoints:"))
        {
            std::cout << "Line:" << __LINE__ << std::endl;
            std::cerr << "Error: Bad input file format" << std::endl;
            exit(1);
        }
        for (int j = 0; j < ModelFitter::NUM_KEYPOINTS_2D; j++) f2 >> net_output_entry[2 * j] >> net_output_entry[2 * j + 1];
        f2 >> str;
        if (str.compare("PAF:"))
        {
            std::cout << "Line:" << __LINE__ << std::endl;
            std::cerr << "Error: Bad input file format" << std::endl;
            exit(1);
        }
        for (int j = 0; j < ModelFitter::NUM_PAF_VEC; j++) f2 >> net_output_entry[3 * j + 2 * ModelFitter::NUM_KEYPOINTS_2D] >> net_output_entry[3 * j + 1 + 2 * ModelFitter::NUM_KEYPOINTS_2D] >> net_output_entry[3 * j + 2 + 2 * ModelFitter::NUM_KEYPOINTS_2D];
        f2 >> net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 0];  // If false, left hand is blurred.
        f2 >> net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 1];  // If false, right hand is blurred.
        f2.close();

        net_output.push_back(net_output_entry); 
    }

    std::vector<std::vector<cv::Point3i>> dense_constraint;

    // initialize total model
    LoadTotalModelFromObj(g_total_model, std::string("model/mesh_nofeet.obj"));
    LoadModelColorFromObj(g_total_model, std::string("model/nofeetmesh_byTomas_bottom.obj"));  // contain the color information
    LoadTotalDataFromJson(g_total_model, std::string("model/adam_v1_plus2.json"), std::string("model/adam_blendshapes_348_delta_norm.json"), std::string("model/correspondences_nofeet.txt"));
    LoadCocoplusRegressor(g_total_model, std::string("model/regressor_0n1_root.json"));

    render->CameraMode(0);
    render->options.K = calibK;
    glutDisplayFunc(emptyfunc);
    glutMainLoopEvent();

    /*
    Stage 1: run single frame fitting & refitting
    */
    auto dense_constraint_entry = dense_constraint.begin();  // unused
    int image_index = FLAGS_start;
    std::vector<smpl::SMPLParams> params;
    // std::vector<CMeshModelInstance> meshes;

    for (auto& net_output_entry: net_output)
    {
        // do not fit hand when hand is blurry
        if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 0] == 0.0)
        {
            std::fill(net_output_entry.data() + 21 * 2, net_output_entry.data() + 42 * 2, 0.0);
            std::fill(net_output_entry.data() + 23 * 3 + 2 * ModelFitter::NUM_KEYPOINTS_2D, net_output_entry.data() + 43 * 3 + 2 * ModelFitter::NUM_KEYPOINTS_2D, 0.0);
        }
        if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 1] == 0.0)
        {
            std::fill(net_output_entry.data() + 42 * 2, net_output_entry.data() + 63 * 2, 0.0);
            std::fill(net_output_entry.data() + 43 * 3 + 2 * ModelFitter::NUM_KEYPOINTS_2D, net_output_entry.data() + 63 * 3 + 2 * ModelFitter::NUM_KEYPOINTS_2D, 0.0);
        }
    }

    if (FLAGS_stage == 1)
    {
        boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal/");

        ModelFitter model_fitter(g_total_model);
        model_fitter.setCalibK(calibK);
        smpl::SMPLParams refit_params;
        refit_params.m_adam_t.setZero();
        refit_params.m_adam_pose.setZero();
        refit_params.m_adam_coeffs.setZero();
        for (auto& net_output_entry: net_output)
        {
            std::cout << "Fitting image " << image_index << " ----------------" << std::endl;

            char basename[200];
            sprintf(basename, "%04d.txt", image_index);
            const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal/" + basename;

            smpl::SMPLParams frame_params;
            frame_params.m_adam_t.setZero();
            frame_params.m_adam_t(2) = 500;
            frame_params.m_adam_pose.setZero();
            frame_params.m_adam_coeffs.setZero();
            frame_params.m_adam_facecoeffs_exp.setZero();

            if (FLAGS_singleStage)
            {
                model_fitter.setFitDataNetOutput(net_output_entry);
                model_fitter.regressor_type = 2;
                model_fitter.fit_face_exp = false;
                model_fitter.euler = false;
                model_fitter.wPosePr = 200.0;
                model_fitter.wCoeffRg = 1.0;
                model_fitter.multiStageFitting();
                model_fitter.readOutParameters(refit_params);
                if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 0] == 0.0)
                {
                    std::copy(params.back().m_adam_pose.data() + 20 * 3, params.back().m_adam_pose.data() + 21 * 3, refit_params.m_adam_pose.data() + 20 * 3);
                    std::copy(params.back().m_adam_pose.data() + 22 * 3, params.back().m_adam_pose.data() + 42 * 3, refit_params.m_adam_pose.data() + 22 * 3);
                }
                if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 1] == 0.0)
                {
                    std::copy(params.back().m_adam_pose.data() + 21 * 3, params.back().m_adam_pose.data() + 22 * 3, refit_params.m_adam_pose.data() + 21 * 3);
                    std::copy(params.back().m_adam_pose.data() + 42 * 3, params.back().m_adam_pose.data() + 62 * 3, refit_params.m_adam_pose.data() + 42 * 3);
                }
            }
            else
            {
                fit_single_frame(g_total_model, net_output_entry.data(), calibK, frame_params, dense_constraint_entry, FLAGS_densepose);

                std::vector<double> reconstruction;
                reconstruct_adam(g_total_model, frame_params, reconstruction);
                refit_single_frame(g_total_model, refit_params, reconstruction, image_index != FLAGS_start, FLAGS_euler);
                if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 0] == 0.0)
                {
                    std::copy(params.back().m_adam_pose.data() + 20 * 3, params.back().m_adam_pose.data() + 21 * 3, refit_params.m_adam_pose.data() + 20 * 3);
                    std::copy(params.back().m_adam_pose.data() + 22 * 3, params.back().m_adam_pose.data() + 42 * 3, refit_params.m_adam_pose.data() + 22 * 3);
                }
                if (image_index != FLAGS_start && net_output_entry[3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 1] == 0.0)
                {
                    std::copy(params.back().m_adam_pose.data() + 21 * 3, params.back().m_adam_pose.data() + 22 * 3, refit_params.m_adam_pose.data() + 21 * 3);
                    std::copy(params.back().m_adam_pose.data() + 42 * 3, params.back().m_adam_pose.data() + 62 * 3, refit_params.m_adam_pose.data() + 42 * 3);
                }
                refit_params.m_adam_facecoeffs_exp = frame_params.m_adam_facecoeffs_exp;
                std::cout << refit_params.m_adam_pose << std::endl;
            }
            params.push_back(refit_params);

            CMeshModelInstance mesh;
            GenerateMesh(mesh, gResultJoint, refit_params, g_total_model, 2, FLAGS_euler);
            // GenerateMesh(mesh, gResultJoint, frame_params, g_total_model, 2, true);
            // meshes.push_back(mesh);

            VisualizedData vis_data;
            CopyMesh(mesh, vis_data);
            render->options.view_dist = gResultJoint[2 * 3 + 2];
            vis_data.vis_type = 2;

            if (image_index == FLAGS_start)
            {
                render->CameraMode(0);
                render->options.K = calibK;
                render->RenderHand(vis_data);
                vis_data.read_buffer = ret_bytes;
                render->RenderAndRead();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            render->CameraMode(0);
            render->options.K = calibK;
            render->RenderHand(vis_data);
            if (FLAGS_OpenGLactive) render->Display();

            vis_data.read_buffer = ret_bytes;
            render->RenderAndRead();

            // convert to opencv format
            cv::Mat frame(ROWS, COLS, CV_8UC4, ret_bytes);
            cv::flip(frame, frame, 0);
            // cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);  // convert to BGR

            sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index);
            const std::string imgName = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
            std::cout << imgName << std::endl;
            cv::Mat img(ROWS, COLS, CV_8UC3, cv::Scalar(0));
            cv::Mat imgr = cv::imread(imgName);
            imgr.copyTo(img.rowRange(0, imgr.rows).colRange(0, imgr.cols));
            cv::Mat aligned = alignMeshImageAlpha(frame, img);
            // cv::Mat aligned = alignMeshImage(frame, cv::imread(imgName));
            sprintf(basename, "%04d.png", image_index);
            const std::string filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal/" + basename;
            assert(cv::imwrite(filename, aligned));

            writeFrameParam(param_filename, refit_params);

            image_index++;
        }
    }

    // perform tracking 
    if (FLAGS_end - FLAGS_start == 1)
    {
        std::cout << "Only one frame, perform no tracking." << std::endl;
        exit(0);
    }

    /*
    Stage 2: Skipped
    */
    std::vector<smpl::SMPLParams> batch_refit_params(net_output.size());
    const int total_frame = FLAGS_end - FLAGS_start;

    /*
    Stage 3: Run tracking
    */
    if (FLAGS_stage <= 3)
    {
        boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal_tracking/");

        ModelFitter model_fitter(g_total_model);
        model_fitter.fit3D = true;
        model_fitter.fit2D = true;
        model_fitter.fitPAF = true;
        model_fitter.fit_surface = true;  // use surface constraints
        model_fitter.euler = false;
        model_fitter.wHandPosePr = 0.1;
        model_fitter.wPosePr = 1.0;
        model_fitter.setCalibK(calibK);
        model_fitter.freezeShape = true;
        for (auto i = 0u, image_index = FLAGS_start + i; i < net_output.size(); i++, image_index++)
        {
            std::cout << "Reading single frame results: " << image_index << std::endl;
            char basename[200];
            sprintf(basename, "%04d.txt", image_index);
            const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal/" + basename;
            smpl::SMPLParams frame_params;
            readFrameParam(param_filename, frame_params);
            batch_refit_params[i] = frame_params;
        }

        CMeshModelInstance mesh0;
        GenerateMesh(mesh0, gResultJoint, batch_refit_params[0], g_total_model, 2, FLAGS_euler);
        VisualizedData vis_data0;
        CopyMesh(mesh0, vis_data0);
        vis_data0.read_depth_buffer = ret_depth;
        vis_data0.read_buffer = ret_bytes;
        render->options.K = calibK;
        render->CameraMode(0);  // ensure the shape of window is correct
        render->RenderDepthMap(vis_data0);  // Render depth map from OpenGL
        render->RenderAndReadDepthMap();
        cv::Mat depthframe0 = cv::Mat(1080, 1920, CV_32F, ret_depth).clone();  // deep copy
        cv::flip(depthframe0, depthframe0, 0);

        char basename[200];
        sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), FLAGS_start);
        const std::string imgName0 = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
        cv::Mat img0(ROWS, COLS, CV_8UC1, cv::Scalar(0));
        cv::Mat img0r = cv::imread(imgName0, CV_LOAD_IMAGE_GRAYSCALE);
        img0r.copyTo(img0.rowRange(0, img0r.rows).colRange(0, img0r.cols));

        // run optical flow && refitting
        for (auto i = 0u, image_index = FLAGS_start + i; i < net_output.size() - 1; i++, image_index++)  // i -> i + 1
        {
            std::cout << "Run tracking image " << image_index << " -> " << image_index + 1 << std::endl;
            sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index);
            const std::string imgName1 = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
            cv::Mat img1(ROWS, COLS, CV_8UC1, cv::Scalar(0));
            cv::Mat img1r = cv::imread(imgName1, CV_LOAD_IMAGE_GRAYSCALE);
            img1r.copyTo(img1.rowRange(0, img1r.rows).colRange(0, img1r.cols));
            sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index + 1);
            const std::string imgName2 = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
            cv::Mat img2(ROWS, COLS, CV_8UC1, cv::Scalar(0));
            cv::Mat img2r = cv::imread(imgName2, CV_LOAD_IMAGE_GRAYSCALE);
            img2r.copyTo(img2.rowRange(0, img2r.rows).colRange(0, img2r.cols));
            cv::Mat img3(ROWS, COLS, CV_8UC1, cv::Scalar(0));
            CMeshModelInstance mesh1, mesh2, mesh3;
            GenerateMesh(mesh1, gResultJoint, batch_refit_params[i], g_total_model, 2, FLAGS_euler);
            if (i < net_output.size() - 2)
            {
                sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index + 2);
                const std::string imgName3 = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
                cv::Mat img3r = cv::imread(imgName3, CV_LOAD_IMAGE_GRAYSCALE);
                img3r.copyTo(img3.rowRange(0, img3r.rows).colRange(0, img3r.cols));
                GenerateMesh(mesh3, gResultJoint, batch_refit_params[i + 2], g_total_model, 2, FLAGS_euler);
            }

            // get the skeleton of the previous frame
            std::vector<double> reconstruction;
            reconstruct_adam(g_total_model, batch_refit_params[i], reconstruction, FLAGS_euler);
            std::fill(reconstruction.data() + 20 * 3, reconstruction.data() + 20 * 3 + 40 * 3, 0.0);  // no smoothing on fingers.
            model_fitter.setFitDataReconstruction(reconstruction);  // joints should be close to the previous frame (in z direction).

            model_fitter.setFitDataNetOutput(net_output[i + 1],
                false,  // body
                false,  // foot
                false,  // hand
                true,  // face
                true  // PAF
            );

            batch_refit_params[i + 1].m_adam_coeffs = batch_refit_params[i].m_adam_coeffs;

            cv::Mat resultMeshImage;
            // for (int t = 0; t < 1; t++)  // iterations for texture mapping
            for (int t = 0; t < 3; t++)  // iterations for texture mapping
            {
                std::vector<cv::Point3i> surface_constraint;
                // generate mesh & depth map
                mesh2.clearMesh();
                GenerateMesh(mesh2, gResultJoint, batch_refit_params[i + 1], g_total_model, 2, FLAGS_euler);  // generate mesh2 here because it is updating every iteration
                VisualizedData vis_data2, vis_data1;
                vis_data2.read_depth_buffer = ret_depth;
                vis_data2.read_buffer = ret_bytes;
                vis_data1.read_depth_buffer = ret_depth;
                vis_data1.read_buffer = ret_bytes;
                CopyMesh(mesh2, vis_data2);
                CopyMesh(mesh1, vis_data1);
                render->CameraMode(0);  // ensure the shape of window is correct
                render->options.K = calibK;
                render->RenderDepthMap(vis_data2);  // Render depth map from OpenGL
                render->RenderAndReadDepthMap();
                cv::Mat depthframe2 = cv::Mat(1080, 1920, CV_32FC1, ret_depth).clone();  // deep copy
                cv::flip(depthframe2, depthframe2, 0);
                render->CameraMode(0);  // ensure the shape of window is correct
                render->options.K = calibK;
                render->RenderDepthMap(vis_data1);  // Render depth map from OpenGL
                render->RenderAndReadDepthMap();
                cv::Mat depthframe1 = cv::Mat(1080, 1920, CV_32FC1, ret_depth).clone();  // deep copy
                cv::flip(depthframe1, depthframe1, 0);

                // run optical flow (forward)
                cv::Mat virtualImage;
                // getVirtualImageConstraint(render, calibK, mesh0, mesh2, depthframe0, depthframe2, img0, img2, virtualImage, surface_constraint, 0);
                getVirtualImageConstraint(render, calibK, mesh1, mesh2, depthframe1, depthframe2, img1, img2, virtualImage, surface_constraint, 0);
                std::cout << "Constraints from texture Optical flow: " << surface_constraint.size() << std::endl;

                // if (FLAGS_imageOF && t == 0)
                if (FLAGS_imageOF)
                {
                    // correspondance from image optical flow
                    std::vector<cv::Point3i> image_surface_constraint;
                    Tracking_MeshVertex_depthMap(true, img1, img2, depthframe1, calibK, mesh1.m_vertices, image_surface_constraint, 0);

                    if (i < net_output.size() - 2)  // also get t + 2 -> t + 1
                    {
                        VisualizedData vis_data3;
                        vis_data3.read_depth_buffer = ret_depth;
                        vis_data3.read_buffer = ret_bytes;
                        CopyMesh(mesh3, vis_data3);
                        render->CameraMode(0);  // ensure the shape of window is correct
                        render->options.K = calibK;
                        render->RenderDepthMap(vis_data3);  // Render depth map from OpenGL
                        render->RenderAndReadDepthMap();
                        depthframe1 = cv::Mat(1080, 1920, CV_32F, ret_depth).clone();  // deep copy
                        cv::flip(depthframe1, depthframe1, 0);

                        std::vector<cv::Point3i> image_surface_constraint2;
                        Tracking_MeshVertex_depthMap(true, img3, img2, depthframe1, calibK, mesh3.m_vertices, image_surface_constraint2, 0);

                        std::array<cv::Point2i, TotalModel::NUM_VERTICES> surface_constraint2_by_vertex;
                        surface_constraint2_by_vertex.fill(cv::Point2i(-1, -1));
                        for (auto& c: image_surface_constraint2)
                            surface_constraint2_by_vertex[c.z] = cv::Point2i(c.x, c.y);
                        const float thresh = 20.0;
                        for (auto i = 0u; i < image_surface_constraint.size(); i++)
                        {
                            const cv::Point2i& point2 = surface_constraint2_by_vertex[image_surface_constraint[i].z];
                            const int x2 = point2.x;
                            const int y2 = point2.y;
                            const float dist = sqrt((image_surface_constraint[i].x - x2) * (image_surface_constraint[i].x - x2) + 
                                                    (image_surface_constraint[i].y - y2) * (image_surface_constraint[i].y - y2));
                            if ((dist > thresh && x2 >= 0 && y2 >= 0) || (x2 < 0 && y2 < 0))
                            {
                                image_surface_constraint.erase(image_surface_constraint.begin() + i);
                                i--;   // Important! This element is erased from the vector, following index decreases.
                            }
                        }
                    }

                    std::array<bool, TotalModel::NUM_VERTICES> covered; covered.fill(false);
                    for (auto& c: surface_constraint)
                    {
                        covered[c.z] = true;
                    }
                    int count = 0;
                    for (auto& c: image_surface_constraint)
                    {
                        if (!covered[c.z])  // this vertex is not covered
                        {
                            surface_constraint.emplace_back(c);
                            count++;
                        }
                    }
                    std::cout << "Constraints added from image Optical flow: " << count << std::endl;
                }

                downSampleConstraints(surface_constraint, 5);

                std::vector<cv::Point3i> surface_constraint_0;  // texture mapped from the first frame
                getVirtualImageConstraint(render, calibK, mesh0, mesh2, depthframe0, depthframe2, img0, img2, virtualImage, surface_constraint_0, 0);
                downSampleConstraints(surface_constraint_0, 20);
                surface_constraint.insert(surface_constraint.end(), surface_constraint_0.begin(), surface_constraint_0.end());

                if (FLAGS_freeze > 0)
                {
                    for (auto i = 0u; i < surface_constraint.size(); i++)
                    {
                        // below hips has color (x, 0, 0)
                        // betweem hips has color (0, y, y)
                        // some (x, y, y) in between
                        const int iv = surface_constraint[i].z;
                        if ((FLAGS_freeze == 1 && g_total_model.m_colors(iv, 0) > 0)
                            || (FLAGS_freeze == 2 && (g_total_model.m_colors(iv, 0) > 0 || g_total_model.m_colors(iv, 1) > 0)))
                        {
                            surface_constraint.erase(surface_constraint.begin() + i);
                            i--;
                        }
                    }
                }

                std::cout << "Constraints from Optical flow: " << surface_constraint.size() << std::endl;

                model_fitter.setSurfaceConstraints2D(surface_constraint);
                model_fitter.initParameters(batch_refit_params[i + 1]);
                model_fitter.resetFitData();
                model_fitter.resetCostFunction();
                for (auto i = 0; i < model_fitter.pCostFunction[0]->m_nCorrespond_adam2joints; i++)
                {
                    model_fitter.pCostFunction[0]->m_targetPts_weight[5 * i + 0] = 0;
                    model_fitter.pCostFunction[0]->m_targetPts_weight[5 * i + 1] = 0;
                    model_fitter.pCostFunction[0]->m_targetPts_weight[5 * i + 2] = 0.5;
                    model_fitter.pCostFunction[0]->m_targetPts_weight[5 * i + 3] = 0;
                    model_fitter.pCostFunction[0]->m_targetPts_weight[5 * i + 4] = 0;
                }
                for (auto i = 8; i < 8 + g_total_model.m_correspond_adam2face70_adamIdx.rows(); i++)  // weight for face keypoints
                {
                    for (auto d = 0; d < 5; d++)
                        model_fitter.pCostFunction[0]->m_targetPts_weight[5 * (i + model_fitter.pCostFunction[0]->m_nCorrespond_adam2joints) + d] = 10. / 70;
                }
                std::fill(model_fitter.pCostFunction[0]->PAF_weight.data(),
                          model_fitter.pCostFunction[0]->PAF_weight.data() + 14,
                          5.);
                std::fill(model_fitter.pCostFunction[0]->PAF_weight.data() + 23,
                          model_fitter.pCostFunction[0]->PAF_weight.data() + model_fitter.pCostFunction[0]->PAF_weight.size(),
                          1.);
                uint count = 0;
                for (auto i = 8 + g_total_model.m_correspond_adam2face70_adamIdx.rows(); i < model_fitter.pCostFunction[0]->m_nCorrespond_adam2pts; i++)  // weight for surface
                {
                    count++;
                    for (auto d = 0; d < 5; d++)
                        model_fitter.pCostFunction[0]->m_targetPts_weight[5 * (i + model_fitter.pCostFunction[0]->m_nCorrespond_adam2joints) + d] = 100. / surface_constraint.size();
                }
                assert(count == surface_constraint.size());
                model_fitter.runFitting();

                model_fitter.readOutParameters(batch_refit_params[i + 1]);
                mesh2.clearMesh();  // clear out the mesh before re-generating
                GenerateMesh(mesh2, gResultJoint, batch_refit_params[i + 1], g_total_model, 2, FLAGS_euler);

            }

            if (net_output[i + 1][3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 0] == 0.0)
            {
                std::copy(batch_refit_params[i].m_adam_pose.data() + 20 * 3, batch_refit_params[i].m_adam_pose.data() + 21 * 3, batch_refit_params[i + 1].m_adam_pose.data() + 20 * 3);
                std::copy(batch_refit_params[i].m_adam_pose.data() + 22 * 3, batch_refit_params[i].m_adam_pose.data() + 42 * 3, batch_refit_params[i + 1].m_adam_pose.data() + 22 * 3);
            }
            if (net_output[i + 1][3 * ModelFitter::NUM_PAF_VEC + 2 * ModelFitter::NUM_KEYPOINTS_2D + 1] == 0.0)
            {
                std::copy(batch_refit_params[i].m_adam_pose.data() + 21 * 3, batch_refit_params[i].m_adam_pose.data() + 22 * 3, batch_refit_params[i + 1].m_adam_pose.data() + 21 * 3);
                std::copy(batch_refit_params[i].m_adam_pose.data() + 42 * 3, batch_refit_params[i].m_adam_pose.data() + 62 * 3, batch_refit_params[i + 1].m_adam_pose.data() + 42 * 3);
            }

            mesh2.clearMesh();  // clear out the mesh before re-generating
            GenerateMesh(mesh2, gResultJoint, batch_refit_params[i + 1], g_total_model, 2, FLAGS_euler);
            VisualizedData vis_data2_;
            vis_data2_.resultJoint = gResultJoint;
            vis_data2_.read_buffer = ret_bytes;
            CopyMesh(mesh2, vis_data2_);

            Renderer::use_color_fbo = true;
            render->CameraMode(0);
            render->options.K = calibK;

            render->RenderHand(vis_data2_);  // Render depth map from OpenGL
            render->RenderAndRead();
            resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
            cv::flip(resultMeshImage, resultMeshImage, 0);

            cv::Mat img2c(ROWS, COLS, CV_8UC3, cv::Scalar(0));
            cv::Mat img2cr = cv::imread(imgName2);
            img2cr.copyTo(img2c.rowRange(0, img2cr.rows).colRange(0, img2cr.cols));
            cv::Mat aligned = alignMeshImageAlpha(resultMeshImage, img2c);

            sprintf(basename, "%04d.png", image_index + 1);
            const std::string filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal_tracking/" + basename;
            assert(cv::imwrite(filename, aligned));

            sprintf(basename, "%04d.txt", image_index + 1);
            const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal_tracking/" + basename;
            writeFrameParam(param_filename, batch_refit_params[i + 1]);

            if (i == 0u)
            {
                VisualizedData vis_data1;
                vis_data1.resultJoint = gResultJoint;
                vis_data1.read_buffer = ret_bytes;
                CopyMesh(mesh1, vis_data1);

                Renderer::use_color_fbo = true;
                render->CameraMode(0);
                render->options.K = calibK;

                render->RenderHand(vis_data1);  // Render depth map from OpenGL
                render->RenderAndRead();
                resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
                cv::flip(resultMeshImage, resultMeshImage, 0);
                // cv::cvtColor(resultMeshImage, resultMeshImage, cv::COLOR_RGBA2BGR);  // convert to BGR

                cv::Mat img1c(ROWS, COLS, CV_8UC3, cv::Scalar(0));
                cv::Mat img1cr = cv::imread(imgName1);
                img1cr.copyTo(img1c.rowRange(0, img1cr.rows).colRange(0, img1cr.cols));
                cv::Mat aligned = alignMeshImageAlpha(resultMeshImage, img1c);

                sprintf(basename, "%04d.png", image_index);
                const std::string filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal_tracking/" + basename;
                assert(cv::imwrite(filename, aligned));

                sprintf(basename, "%04d.txt", image_index);
                const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/body_3d_frontal_tracking/" + basename;
                writeFrameParam(param_filename, batch_refit_params[i]);
            }
        }
    }
}