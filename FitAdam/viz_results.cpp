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

DEFINE_string(root_dirs, "", "Base root folder to access data");
DEFINE_string(seqName, "default", "Sequence Name to run");
DEFINE_string(resName, "body_3d_frontal_tracking", "Name of the dirctory containing total capture results to visualize.");
DEFINE_bool(noTrack, false, "Visualize output before tracking refinement.");
DEFINE_int32(start, 1, "Starting frame");
DEFINE_int32(end, 1000, "Ending frame");

TotalModel g_total_model;
const int NUM_JOINT_PARAMS = 21 * 3 + 2 * 21 * 3;
double gResultJoint[NUM_JOINT_PARAMS];
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
    std::cout << "Results Name: " << FLAGS_resName << std::endl;
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

int main(int argc, char* argv[])
{
    check_flags(argc, argv);
    render.reset(new Renderer(&argc, argv));  // initialize the OpenGL renderer
    render->options.meshSolid = true;
    render->options.show_joint = false;
    Renderer::use_color_fbo = true;

    // read in camera data
    double calibK[9];  // K Matrix
    const std::string calib_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/calib.json";
    Json::Value json_root;
    std::ifstream f(calib_filename.c_str());
    if (!f.good()) {
        std::cerr << "Error: Calib file " << calib_filename << " does not exists" << std::endl;
        exit(1);
    }
    f >> json_root;
    calibK[0] = json_root["K"][0u][0u].asDouble(); calibK[1] = json_root["K"][0u][1u].asDouble(); calibK[2] = json_root["K"][0u][2u].asDouble(); calibK[3] = json_root["K"][1u][0u].asDouble(); calibK[4] = json_root["K"][1u][1u].asDouble(); calibK[5] = json_root["K"][1u][2u].asDouble(); calibK[6] = json_root["K"][2u][0u].asDouble(); calibK[7] = json_root["K"][2u][1u].asDouble(); calibK[8] = json_root["K"][2u][2u].asDouble();
    f.close();

    // initialize total model
    LoadTotalModelFromObj(g_total_model, std::string("model/mesh_nofeet.obj"));
    LoadModelColorFromObj(g_total_model, std::string("model/nofeetmesh_byTomas_bottom.obj"));  // contain the color information
    LoadTotalDataFromJson(g_total_model, std::string("model/adam_v1_plus2.json"), std::string("model/adam_blendshapes_348_delta_norm.json"), std::string("model/correspondences_nofeet.txt"));
    LoadCocoplusRegressor(g_total_model, std::string("model/regressor_0n1_root.json"));

    render->CameraMode(0);
    render->options.K = calibK;
    glutDisplayFunc(emptyfunc);
    glutMainLoopEvent();

    // read in fitting results for all frames
    std::string resDir = "/" + FLAGS_resName + "/";
    int numFrames = FLAGS_end - FLAGS_start;
    std::vector<smpl::SMPLParams> modelParams(numFrames);
    for (auto i = 0u, image_index = FLAGS_start + i; i < numFrames; i++, image_index++) {
        std::cout << "Reading single frame results: " << image_index << std::endl;
        char basename[200];
        sprintf(basename, "%04d.txt", image_index);
        const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + resDir + basename;
        smpl::SMPLParams frame_params;
        readFrameParam(param_filename, frame_params);
        modelParams[i] = frame_params;
    }

    // TODO read in floor parameters

    // go through each frame and render
    // boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/front_renders/");
    std::string namePre = FLAGS_resName + "_";
    boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "front_renders/");
    boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "joint_front_renders/");
    boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "joint_side_renders/");
    boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "side_renders/");
    boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "top_renders/");
    // first pass to determine floor height
    CMeshModelInstance mesh;
    std::vector<double> maxyHeights(numFrames); 
    double maxOff = std::numeric_limits<double>::min();
    double frontOff;
    for (auto i = 0u, image_index = FLAGS_start + i; i < numFrames; i++, image_index++) {
        mesh.clearMesh();
        GenerateMesh(mesh, gResultJoint, modelParams[i], g_total_model, 2, false); // use axis-angle
        // find min y value
        /*for (int j = 1; j < NUM_JOINT_PARAMS; j += 3) {
             if (gResultJoint[j] < miny) {
                 miny = gResultJoint[j];
		 std::cout << miny << std::endl;
             }
        }*/
        double maxy = std::numeric_limits<double>::min(); // actually have to find max rather than min b/c mesh is flipped
        for (int j = 0; j < TotalModel::NUM_VERTICES; j++)  //Vertices
        {
            double cury = (double)mesh.m_vertices[j].y; //+ modelParams[i].m_adam_t[1];
            if (cury > maxy) {
                maxy = cury;
            }
        }
        maxyHeights[i] = maxy;

        double curOff = gResultJoint[2 * 3 + 2];
        if (curOff > maxOff) {
            maxOff = curOff;
        }

        if (i == 0) {
            frontOff = curOff;
        }
    }

    // floor height will be median of sorted max verts
    std::sort(maxyHeights.begin(), maxyHeights.end());
    double floorHeight;
    if (numFrames % 2 == 1) {
        floorHeight = maxyHeights[(maxyHeights.size()-1) / 2];
    } else {
        int idx1 = maxyHeights.size() / 2;
        int idx2 = idx1 - 1;
        floorHeight = (maxyHeights[idx1] + maxyHeights[idx2]) / 2;
    }

    // now actually render
    cv::Mat resultMeshImage;
    for (auto i = 0u, image_index = FLAGS_start + i; i < numFrames; i++, image_index++) {
        std::cout << "Rendering frame " << image_index << std::endl;
        char basename[200];
        sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index);
        const std::string imgName2 = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
        // get the mesh
        mesh.clearMesh();
        GenerateMesh(mesh, gResultJoint, modelParams[i], g_total_model, 2, false, false); // use axis-angle
        sprintf(basename, "%04d.png", image_index);
        
        VisualizedData vis_data2_;
        vis_data2_.resultJoint = gResultJoint;
        vis_data2_.read_buffer = ret_bytes;
        CopyMesh(mesh, vis_data2_);
        render->options.view_dist = frontOff; //gResultJoint[2 * 3 + 2];
        vis_data2_.vis_type = 2;
        std::array<double, 3> groundNormal = {0.0, -1.0, 0.0};
        std::array<double, 3> groundCenter = {0.0, floorHeight, 0.0};
        vis_data2_.ground_normal = groundNormal;
        vis_data2_.ground_center = groundCenter;
        render->options.zmax = 3000.0f;
        render->options.show_shadows = true;

        /*
         FRONT VIEW
         */
        Renderer::use_color_fbo = true;
        vis_data2_.bRenderFloor = true;
        render->CameraMode(0); // front view
        render->options.K = calibK;

        render->RenderHand(vis_data2_);  // Render depth map from OpenGL
        render->RenderAndRead();
        resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
        cv::flip(resultMeshImage, resultMeshImage, 0);

        // cv::Mat img2c(ROWS, COLS, CV_8UC3, cv::Scalar(0));
        // cv::Mat img2cr = cv::imread(imgName2);
        // img2cr.copyTo(img2c.rowRange(0, img2cr.rows).colRange(0, img2cr.cols));
        // cv::Mat aligned = alignMeshImageAlpha(resultMeshImage, img2c);

        sprintf(basename, "%04d.png", image_index);
        const std::string frontFilename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "front_renders/" + basename;
        assert(cv::imwrite(frontFilename, resultMeshImage));
        std::cout << "Finished front view!" << std::endl;
        vis_data2_.bRenderFloor = false;

        /*
        FRONT VIEW with joints
         */
        Renderer::use_color_fbo = true;
        render->options.show_joint = true;
        render->options.show_mesh = false;
        vis_data2_.bRenderFloor = true;
        render->CameraMode(0); // front view
        render->options.K = calibK;

        render->RenderHand(vis_data2_);  // Render depth map from OpenGL
        render->RenderAndRead();
        resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
        cv::flip(resultMeshImage, resultMeshImage, 0);

        const std::string jointFilename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "joint_front_renders/" + basename;
        assert(cv::imwrite(jointFilename, resultMeshImage));
        std::cout << "Finished joint front view!" << std::endl;
        render->options.show_joint = false;
        render->options.show_mesh = true;
        vis_data2_.bRenderFloor = false;

        render->options.view_dist = maxOff; // no other views are from front

        /*
         SIDE VIEW
         */
        Renderer::use_color_fbo = true;
	    vis_data2_.bRenderFloor = true;
        render->CameraMode(2); // side view
        render->options.K = calibK;

        render->RenderHand(vis_data2_);  // Render depth map from OpenGL
        render->RenderAndRead();
        resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
        cv::flip(resultMeshImage, resultMeshImage, 0);

        const std::string sideFilename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "side_renders/" + basename;
        assert(cv::imwrite(sideFilename, resultMeshImage));
        std::cout << "Finished side view!" << std::endl;
        vis_data2_.bRenderFloor = false;

        /*
        SIDE VIEW with joints
         */
        Renderer::use_color_fbo = true;
        render->options.show_joint = true;
        render->options.show_mesh = false;
        vis_data2_.bRenderFloor = true;
        render->CameraMode(2); // side view
        render->options.K = calibK;

        render->RenderHand(vis_data2_);  // Render depth map from OpenGL
        render->RenderAndRead();
        resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
        cv::flip(resultMeshImage, resultMeshImage, 0);

        const std::string jointSideFilename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "joint_side_renders/" + basename;
        assert(cv::imwrite(jointSideFilename, resultMeshImage));
        std::cout << "Finished joint side view!" << std::endl;
        render->options.show_joint = false;
        render->options.show_mesh = true;
        vis_data2_.bRenderFloor = false;
 
        /*
         TOP VIEW
         */
        Renderer::use_color_fbo = true;
        vis_data2_.bRenderFloor = true;
        render->CameraMode(1); // top view
        render->options.K = calibK;

        render->RenderHand(vis_data2_);  // Render depth map from OpenGL
        render->RenderAndRead();
        resultMeshImage = cv::Mat(1080, 1920, CV_8UC4, ret_bytes).clone();  // deep copy
        cv::flip(resultMeshImage, resultMeshImage, 0);

        const std::string topFilename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/" + namePre + "top_renders/" + basename;
        assert(cv::imwrite(topFilename, resultMeshImage));
        std::cout << "Finished top view!" << std::endl;
        vis_data2_.bRenderFloor = false;

    }

}
