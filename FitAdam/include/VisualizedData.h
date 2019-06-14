#ifndef VISUALIZEDATA
#define VISUALIZEDATA

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <GL/glut.h>

enum EnumRenderType
{
	RENDER_None=-1,
	RENDER_Play_Frames=0,
	RENDER_Play_CamMove=1,			//3DPS
};

class VisualizedData
{
public:
	VisualizedData()
	{
		//Init color Map
		cv::Mat colorMapSource = cv::Mat::zeros(256,1,CV_8U);
		for(unsigned int i=0;i<=255;i++)
			colorMapSource.at<uchar>(i,0) = i;
		cv::Mat colorMap;
		cv::applyColorMap(colorMapSource, colorMap, cv::COLORMAP_JET);
		for(unsigned int i=0;i<=255;i++)
		{
			cv::Point3f tempColor;
			tempColor.z = colorMap.at<cv::Vec3b>(i,0)[0]/255.0; //blue
			tempColor.y = colorMap.at<cv::Vec3b>(i,0)[1]/255.0; //green
			tempColor.x = colorMap.at<cv::Vec3b>(i,0)[2]/255.0;	//red
			m_colorMapGeneral.push_back(tempColor);
		}

		m_backgroundColor = cv::Point3f(1,1,1);
		m_pPickedBoneGroup = NULL;

		//Set frame as local part's frame
		for(int i=0;i<16;++i)
			m_AnchorMatrixGL[i]=0;
		m_AnchorMatrixGL[0] = 1;
		m_AnchorMatrixGL[5] = 1;
		m_AnchorMatrixGL[10] = 1;
		m_AnchorMatrixGL[15] = 1;
		
		bShowBackgroundTexture = false;
		bRenderFloor = false;
		m_renderType=RENDER_None;
		m_selectedHandIdx = std::make_pair(0,0);
		//g_shaderProgramID =0;
		//m_reloadShader = false;
		read_buffer = NULL;
		read_depth_buffer = nullptr;
		targetJoint = NULL;
		resultJoint = NULL;
		vis_type = 0;

		int hand[] = {0, 1, 1, 2, 2, 3, 3, 4,
			0, 5, 5, 6, 6, 7, 7, 8,
			0, 9, 9, 10, 10, 11, 11, 12,
			0, 13, 13, 14, 14, 15, 15, 16,
			0, 17, 17, 18, 18, 19, 19, 20
		};
		std::vector<int> connMat_hand(hand, hand + sizeof(hand) / sizeof(int));
		connMat.push_back(connMat_hand);

		int body[] = {
			0, 1,
			0, 3, 3, 4, 4, 5,
			0, 9, 9, 10, 10, 11,
			0, 2,
			2, 6, 6, 7, 7, 8,
			2, 12, 12, 13, 13, 14,
			1, 15, 15, 16,
			1, 17, 17, 18
		};
		std::vector<int> connMat_body(body, body + sizeof(body) / sizeof(int));
		connMat.push_back(connMat_body);

		std::vector<int> connMat_total(body, body + sizeof(body) / sizeof(int));
		std::vector<int> connMat_lhand(0);
		std::vector<int> connMat_rhand(0);
		for (auto i = 0u; i < connMat_hand.size(); i++)
		{
			connMat_total.push_back(connMat_hand[i] + 21); // left hand
			connMat_lhand.push_back(connMat_hand[i] + 21);
		}
		for (auto i = 0u; i < connMat_hand.size(); i++)
		{
			connMat_total.push_back(connMat_hand[i] + 42); // right hand
			connMat_rhand.push_back(connMat_hand[i] + 42);
		}
		connMat.push_back(connMat_total);
		connMat.push_back(connMat_lhand);
		connMat.push_back(connMat_rhand);
	}

	void clear()
	{
		m_meshVertices.clear();
		m_meshVerticesColor.clear();
		m_meshVerticesNormal.clear();
		m_meshVerticesUV.clear();
		m_meshVerticesAlpha.clear();
		m_meshIndices.clear();
	}

	~VisualizedData() {}
	
	// std::vector<CamVisInfo> m_camVisVector;
	// std::vector<textureCoord> m_camTexCoord;
	// bool m_loadNewCamTextureTrigger;
	// std::vector<cv::Mat> m_camTextureImages;	//Num should be the same as m_camVisVector
	// std::vector<CamVisInfo> m_newlyRegisteredCamVisVector;
	// std::vector<cv::Point3f> m_cameraColorVectorByMotionCost;
	// std::vector<cv::Point3f> m_cameraColorVectorByNormalCost;
	// std::vector<cv::Point3f> m_cameraColorVectorByAppearanceCost;
	// std::vector<cv::Point3f> m_cameraColorVectorByTotalDataCost;

	//Patch Clound & Trajectory Stream
	// std::vector<PatchCloudUnit> m_patchCloud;
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_trajectoryTotal;  //line
	// std::vector< float> m_trajectoryTotal_alpha;  //line

	//Mesh Structure;
	std::vector<cv::Point3f> m_meshVertices_f;
	std::vector<cv::Point3d> m_meshVertices;		
	std::vector<cv::Point3d> m_meshVerticesColor;		
	std::vector<cv::Point3d> m_meshVerticesNormal;
	std::vector<cv::Point2d> m_meshVerticesUV;
	std::vector<double> m_meshVerticesAlpha;
	std::vector<unsigned int> m_meshIndices;
	std::vector< std::pair<cv::Point3f, cv::Point3f> > m_meshJoints;		//point, color

	// std::vector<CSkeletonVisUnit> m_skeletonVisVector;
	// std::vector<CSkeletonVisUnit> m_skeletonVisVectorCompare;
	void* m_pPickedBoneGroup;		//type is CPartTrajProposal*
	GLfloat m_AnchorMatrixGL[16]; //To make normalize coordinate w.r.t the selected bone

	//Face
	std::vector< cv::Point3d > m_faceCenters; //point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_faceLandmarks; //point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_faceNormal;//point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_faceLandmarksGT; //point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_faceNormalGT;//point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_faceLandmarks_pm; //point,color
	std::vector<std::pair<cv::Point3d, std::string > > m_faceNames_pm;
	GLfloat m_face_pm_modelViewMatGL[16];
	// std::vector< CFaceVisUnit > m_faceAssociated;

	//SSP 
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_gazePrediction; //point,color


	//Hands
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_handLandmarks; //point,color
	std::vector< std::pair<cv::Point3d,cv::Point3d> > m_handNormal;//point,color
	GLfloat m_hand_modelViewMatGL[16];
	std::pair<int,int> m_selectedHandIdx;


	//Foot
	std::vector< std::pair<cv::Point3d, cv::Point3d> > m_footLandmarks; //point,color
	std::vector< std::pair<cv::Point3d, cv::Point3d> > m_footLandmarks_ankleBone; //point,color

	//Gaze Engagement
	std::vector< std::pair<cv::Point3d, cv::Point3d> > m_gazeEngInfo;

	//Visual Hull Generation
	// CVolumeVisUnit m_visualHullVisualizeUnit;
	// CVolumeVisUnit m_faceNodeProposalVisUnit;
	// CVolumeVisUnit m_nodeProposalVisUnit;
	
	// //General purpose component
	// std::vector< VisStrUnit> m_debugStr; //point,color
	// std::vector< std::pair<cv::Point3d, std::pair< cv::Point3d, float> > > m_debugLinesWithAlpha; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_superTrajLines; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugLines; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugPt; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugSphere; //point,color
	// std::vector< std::pair<Bbox3D,cv::Point3d> > m_debugCubes;
	// //I made an additional layer to use "debug display tools"
	// //Instead of addition directly to (for example) m_debugPt,
	// //Add data to m_debugPtData, which draw it through m_debugPt, in SFMManager.visualizeEvertyhing() function
	// //I do this 1) m_debugPt is cleared in every visualization, so need to add there in every step again
	// //			2) m_debugPt can be used in  other data structure for simple visualization 
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugLineData; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugPtData; //point,color
	// std::vector< std::pair<cv::Point3d,cv::Point3d> > m_debugSphereData; //point,color
	// std::vector< VisStrUnit> m_debugStrData; //point,color

	//Rendering and Save to Images
	// bool m_saveToFileTrigger;
	EnumRenderType m_renderType;		
	// void SaveToImage(bool bCamView=false,bool bFrameIdxAsName=false);

	cv::Point3f m_backgroundColor;	//opengl background color

	// //Texture of Patch Cloud. Not used anymore
	// std::vector<textureCoord> g_trajTextCoord;
	// std::vector<cv::Mat*> m_PatchVector;

	std::vector< cv::Point3f > m_colorMapGeneral;    //0~255, 0:bluish,    255:reddish   0<colorValueRGB <1 in order to be used in OpenGL
	// cv::Point3f GetColorByCost(float cost,float minCost,float maxCost);

	//background texture
	//bool bReLoadBackgroundTexture;
	bool bShowBackgroundTexture;
	bool bRenderFloor;

	double* targetJoint;
	double* resultJoint;
	uint vis_type; // 0 for hand, 1 for body, 2 for body with hands
	std::vector<std::vector<int>> connMat;

	//Shader
	//bool m_reloadShader;
	//GLuint m_shaderProgramID;
	GLubyte* read_buffer;
	float* read_depth_buffer;
	float* read_rgbfloat_buffer;
	std::array<double, 3> ground_center;
	std::array<double, 3> ground_normal;
	cv::Mat backgroundImage;
};

struct VisualizationOptions
{
	VisualizationOptions(): K(NULL), xrot(0.0f), yrot(0.0f), view_dist(300.0f), nRange(40.0f), CameraMode(0u), show_joint(true), show_mesh(true),
		ortho_scale(1.0f), width(600), height(600), zmin(0.01f), zmax(1000.0f), meshSolid(false) {}
	double* K;
	GLfloat	xrot, yrot;
	GLfloat view_dist, nRange;
	uint CameraMode;
	float ortho_scale;
	GLint width, height;
	GLfloat zmin, zmax; // used only in camera mode (to determine the range of objects in z direction)
	bool meshSolid;
	bool show_joint;
	bool show_mesh;
};

#endif