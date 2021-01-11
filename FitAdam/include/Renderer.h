#pragma once
#include "VisualizedData.h"
#include <string>
#include <opencv2/opencv.hpp>

// using namespace adam
// {
class Renderer
{
public:
	Renderer(int* argc, char** argv);
	void RenderHand(VisualizedData& g_visData);
	void RenderHandSimple(VisualizedData& g_visData);
	void RenderDepthMap(VisualizedData& g_visData);
	void RenderProjection(VisualizedData& g_visData);
	// static void IdleSaveImage();
	static void RenderAndRead();
	static void RenderAndReadDepthMap();
	static void RenderAndReadProjection();
	void Display();
	static VisualizationOptions options;
	void NormalMode(uint position=0u, int width=600, int height=600);
	void CameraMode(uint position=0u, int width=1920, int height=1080, double* calibK=NULL);
	void OrthoMode(float scale, uint position=0u);
	static bool use_color_fbo;
private:
	//initialization functions
	void simpleInit(); // my simple way of initialization
	void InitGraphics();
	static const std::string SHADER_ROOT;
	static GLuint g_shaderProgramID[10];
	enum draw_mode {MODE_DRAW_DEFUALT=0, MODE_DRAW_SELECTION_PTCLOUD, MODE_DRAW_DEPTH_RENDER, MODE_DRAW_NORMALMAP, MODE_DRAW_PTCLOUD, MODE_DRAW_MESH, MODE_DRAW_FLOOR, MODE_DRAW_MESH_TEXTURE, MODE_DRAW_PROJECTION, MODE_DRAW_SHADOW};
	void SetShader(const std::string shaderName, GLuint& programId);
	GLuint LoadShaderFiles(const char* vertex_file_path, const char* fragment_file_path, bool verbose=false);

	//global variables
	static int g_drawMode;
	static GLuint g_vao;		//Vertex Array: only need one, before starting opengl window
	static GLuint g_vertexBuffer, g_uvBuffer, g_normalBuffer, g_indexBuffer;	//Vertex Buffer. 
	static GLuint g_groundVAO, g_groundVBO; // for the floor (VBO holds pos, normal, and color)
	static GLuint g_shadowFBO, g_shadowTexture; // for shadows on the floor
	static GLuint g_fbo_color, g_fbo_depth, g_fbo_rgbfloat;		//frame buffer object
	static GLuint g_colorTexture, g_depthTexture, g_rgbfloatTexture, g_imageTexture;
	static GLuint g_depth_renderbuffer;
	static int g_depthRenderViewPortW, g_depthRenderViewPortH;
	static int g_colorTextureBufferWidth, g_colorTextureBufferHeight;
	static int SHADOW_WIDTH, SHADOW_HEIGHT;
	static GLint window_id;

	//pointer to data
	static VisualizedData* pData;

	//Call back functions
	static void reshape(int w, int h);
	// static void CameraModeReshape(int w, int h);
	static void SimpleRenderer();
	static void MeshRender();
	static void DepthMapRenderer();
	static void DrawSkeleton(double* joints, uint vis_type, std::vector<int> connMat);
	static void DrawFloor();
	static void DrawShadowMap();
	static void SpecialKeys(const int key, const int x, const int y);
	static void IdleReadBuffer();
	static void ProjectionRenderer();
};
// }
