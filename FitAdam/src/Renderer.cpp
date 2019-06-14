#include <GL/glew.h>
#include "Renderer.h"
#include <GL/glut.h>    
#include <GL/freeglut.h>    
#include <glm/glm.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <math.h>
#include <vector>
// #include <FreeImage.h>
#include <opencv2/core/mat.hpp>
#include <assert.h>
#include <Eigen/Dense>
#define PI 3.14159265
// #define OPENGL_DEBUG

const std::string Renderer::SHADER_ROOT = "./Shaders/";
GLuint Renderer::g_shaderProgramID[8];

int Renderer::g_drawMode = 0;
GLuint Renderer::g_vao = 0;     //Vertex Array: only need one, before starting opengl window
GLuint Renderer::g_vertexBuffer, Renderer::g_uvBuffer, Renderer::g_normalBuffer, Renderer::g_indexBuffer;   //Vertex Buffer. 
GLuint Renderer::g_fbo_color, Renderer::g_fbo_depth, Renderer::g_fbo_rgbfloat;        //frame buffer object
GLuint Renderer::g_colorTexture, Renderer::g_depthTexture, Renderer::g_rgbfloatTexture, Renderer::g_imageTexture;
GLuint Renderer::g_depth_renderbuffer;
int Renderer::g_depthRenderViewPortW=640, Renderer::g_depthRenderViewPortH=480;
int Renderer::g_colorTextureBufferWidth=1920, Renderer::g_colorTextureBufferHeight=1080;
bool Renderer::use_color_fbo = false;

GLint Renderer::window_id;

VisualizedData* Renderer::pData = NULL;
VisualizationOptions Renderer::options;

#ifdef OPENGL_DEBUG
void GLAPIENTRY MessageCallback( GLenum source,
                                 GLenum type,
                                 GLuint id,
                                 GLenum severity,
                                 GLsizei length,
                                 const GLchar* message,
                                 const void* userParam )
{
    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}
#endif

Renderer::Renderer(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitWindowSize(options.width, options.height);
    glutInitWindowPosition(200, 0);
    glutInitDisplayMode ( GLUT_RGBA  | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glEnable(GL_MULTISAMPLE);
    window_id = glutCreateWindow("Adam Model Renderer");
    this->InitGraphics();
    this->simpleInit();
}

void Renderer::InitGraphics()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);

    if (glewInit() != GLEW_OK) {
        std::string errorMessage("Failed to initialize GLEW");
        throw std::runtime_error(errorMessage);
    }

    this->SetShader("SimplestShader", g_shaderProgramID[MODE_DRAW_DEFUALT]);
    this->SetShader("SimplestShader", g_shaderProgramID[MODE_DRAW_PTCLOUD]);
    this->SetShader("DepthMapRender", g_shaderProgramID[MODE_DRAW_DEPTH_RENDER]);
    this->SetShader("Selection", g_shaderProgramID[MODE_DRAW_SELECTION_PTCLOUD]);
    this->SetShader("NormalMap", g_shaderProgramID[MODE_DRAW_NORMALMAP]);
    this->SetShader("Mesh", g_shaderProgramID[MODE_DRAW_MESH]);
    this->SetShader("Mesh_texture_shader", g_shaderProgramID[MODE_DRAW_MESH_TEXTURE]);
    this->SetShader("Project2D", g_shaderProgramID[MODE_DRAW_PROJECTION]);

#ifdef OPENGL_DEBUG
    // During init, enable debug output
    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );
#endif

    glUseProgram(0);
    g_drawMode = MODE_DRAW_DEFUALT;

    //The following is only used to draw 3D point cloud (or the trajectory cloud at a time)
    //Vertex Array
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);

    //Vertex buffer
    glGenBuffers(1, &g_vertexBuffer);           //Vertices of pt cloud or mesh
    glGenBuffers(1, &g_uvBuffer);
    glGenBuffers(1, &g_normalBuffer);
    glGenBuffers(1, &g_indexBuffer);            //Mesh face's indices

    //The following is for off-screen rendering
    //Create color frame buffer (used to picking)
    glEnable(GL_TEXTURE_2D);
    glGenFramebuffers(1, &g_fbo_color);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_color);
    glGenTextures(1, &g_colorTexture);
    glBindTexture(GL_TEXTURE_2D, g_colorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_colorTextureBufferWidth, g_colorTextureBufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);        //1920, 1080 is the maximum available windows size
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER,
                         GL_COLOR_ATTACHMENT0,
                         g_colorTexture, 0);
    glGenRenderbuffers(1, &g_depth_renderbuffer);   // a depth buffer is needed for FBO rendering
    glBindRenderbuffer(GL_RENDERBUFFER, g_depth_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1920, 1080);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_depth_renderbuffer);
    if(GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
        printf("Failed in color framebuffer setting\n");
        return;
    }

    //Create depth frame buffer (used to depth rendering)
    glGenFramebuffers(1, &g_fbo_depth);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_depth);
    glGenTextures(1, &g_depthTexture);
    glBindTexture(GL_TEXTURE_2D, g_depthTexture);
    // glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT32, 640, 480, 0,GL_DEPTH_COMPONENT, GL_DOUBLE, 0);
    // glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT32, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_DOUBLE, 0);        //1920, 1080 is the maximum
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    // We're going to read from this, but it won't have mipmaps,
    // so turn off mipmaps for this texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER,
                         GL_DEPTH_ATTACHMENT,
                         g_depthTexture, 0);
    if(GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
        printf("Failed in depth framebuffer setting\n");
        return;
    }

    //Create rgb float frame buffer
    glGenFramebuffers(1, &g_fbo_rgbfloat);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_rgbfloat);
    glGenTextures(1, &g_rgbfloatTexture);
    glBindTexture(GL_TEXTURE_2D, g_rgbfloatTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1920, 1080, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_rgbfloatTexture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_depth_renderbuffer); // also attach the depth renderbuffer to this framebuffer
    if(GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
        printf("Failed in RGFloat framebuffer setting\n");
        exit(0);
    }

    glGenFramebuffers(1, &g_imageTexture);
    
    //Come back to original screen framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER,0);    
    glDisable(GL_TEXTURE_2D);
}

void Renderer::SetShader(const std::string shaderName, GLuint& programId)
{
    std::string vertexShaderFile = SHADER_ROOT + "/" + shaderName + ".vertexshader";
    std::string fragmentShaderFile = SHADER_ROOT + "/" + shaderName + ".fragmentshader";
    programId = this->LoadShaderFiles(vertexShaderFile.c_str(),fragmentShaderFile.c_str());
}

GLuint Renderer::LoadShaderFiles(const char* vertex_file_path, const char* fragment_file_path, bool verbose)
{
    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open()){
        std::string Line = "";
        while(getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }
    else
    {
        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
        return 0;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }

    GLint Result = GL_FALSE;
    int InfoLogLength;

    // Compile Vertex Shader
    if (verbose)
        printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    // Compile Fragment Shader
    if (verbose)
        printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    if (verbose)
        printf("Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> ProgramErrorMessage(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    if (verbose)
        printf("Setting Shader has been done\n");
    return ProgramID;
}

void Renderer::simpleInit()
{
    // Init function
    GLfloat ambientLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };// <1>
    GLfloat diffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };// <2>
    GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };// <3>
    GLfloat specref[] = { 1.0f, 1.0f, 1.0f, 1.0f };// <4>
    GLfloat light0pos[] = { 0.0f, -1.0f, 0.0f, 0.0f };
    glClearColor(0.0, 0.0, 0.0, 1.0);
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light0pos);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
    glMateriali(GL_FRONT, GL_SHININESS, 128);

    glutReshapeFunc(this->reshape);
    glutSpecialFunc(this->SpecialKeys);
}

void Renderer::reshape(int w, int h)
{
    if(h == 0) h = 1;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // this section allows for window reshaping while
    // maintaining a "normal" GLUT shape
    if(options.CameraMode == 0u)
    {
        glFrustum(-options.nRange/2, options.nRange/2, -options.nRange*h/w/2, options.nRange*h/w/2, -options.nRange + options.view_dist, options.nRange + options.view_dist);
    }
    else if(options.CameraMode == 1u)
    {
        // camera model: set the perspective parameters according to K
        assert(options.K != NULL); // The K matrix must have been provided.
        assert(options.zmax > options.zmin);
        // glOrtho(-960, 960, -540, 540, options.zmin, options.zmax);
        // const double l = 0.0, r = options.width, b = options.height, t = 0.0;
        const double l = 0.0, r = w, b = h, t = 0.0;
        const double tx = -(r+l)/(r-l), ty = -(t+b)/(t-b), tz = -(options.zmax+options.zmin)/(options.zmax-options.zmin);
        double ortho[16] = {2./(r-l), 0., 0., tx,
                                  0., 2./(t-b), 0., ty,
                                  0., 0., -2./(options.zmax-options.zmin), tz,
                                  0., 0., 0., 1.};
        double* K = options.K; // aliasing
        double Intrinsics[16] = {
            K[0], K[1], K[2], 0.0,
            K[3], K[4], K[5], 0.0,
            K[6], K[7], -(options.zmin+options.zmax), options.zmin*options.zmax,
            0.0, 0.0, 1.0, 0.0
        };
        cv::Mat orthoMat = cv::Mat(4, 4, CV_64F, ortho);
        cv::Mat IntrinsicMat = cv::Mat(4, 4, CV_64F, Intrinsics);
        cv::Mat Frustrum = orthoMat*IntrinsicMat;
        Frustrum = Frustrum.t();
        GLdouble projMatrix[16];
        for (int i = 0; i < 16; i++) projMatrix[i] = ((double*)Frustrum.data)[i];
        glLoadMatrixd(projMatrix);
    }
    else
    {
        assert(options.CameraMode == 2u);
        glOrtho(-w * options.ortho_scale/2, w * options.ortho_scale/2, -h * options.ortho_scale/2, h * options.ortho_scale/2, -options.view_dist, options.view_dist);
    }
}

void Renderer::RenderHand(VisualizedData& g_visData)
{
    pData = &g_visData;
    glutDisplayFunc(MeshRender);
    
}

void Renderer::RenderHandSimple(VisualizedData& g_visData)
{
    // use point cloud renderer
    pData = &g_visData;
    glutDisplayFunc(SimpleRenderer);
}

void Renderer::SimpleRenderer()
{
    if (use_color_fbo)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_color);
        glViewport(0, 0, options.width, options.height);
        GLenum FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        // GLfloat white[3] = {1., 1., 1.};
        // glClearBufferfv(GL_COLOR, 0, white);
        if (GL_FRAMEBUFFER_COMPLETE != FBOstatus)
        {
            std::cout << "FrameBuffer Fails." << std::endl;
            exit(0);
        }
        static const GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, draw_buffers);
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, options.width, options.height);
    }

    glClearColor(1., 1., 1., 0.);
    // mesh renderer
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();
    if (options.CameraMode == 0u)
    {
        gluLookAt(0, -options.view_dist * sin(0.0), -options.view_dist * cos(0.0), 0, 0, 0, 0, -1, 0);
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
    }
    else if (options.CameraMode == 2u)
    {
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
        const float centerx = 1920 * options.ortho_scale / 2;
        const float centery = 1080 * options.ortho_scale / 2;
        gluLookAt(centerx, centery, 0, centerx, centery, 1, 0, -1, 0);
    }
    else
    {
        assert(options.CameraMode == 1u);
        glTranslatef(0, 0, options.view_dist);
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
        glTranslatef(0, 0, -options.view_dist);
    }

    if (options.CameraMode == 0u)
    {
        cv::Point3d min_s(10000., 10000., 10000.);
        cv::Point3d max_s(-10000., -10000., -10000.);
        assert(pData->resultJoint);
        int start_idx, end_idx;
        if (pData->vis_type <= 2)
        {
            start_idx = 0;
            if (pData->vis_type == 0)  // for hand
                end_idx = 21;
            else if(pData->vis_type == 1)  // for body (SMC order)
                end_idx = 21;
            else if (pData->vis_type == 2) // for hand and body
                end_idx = 62;
        }
        else if(pData->vis_type == 3)
        {
            start_idx = 21;
            end_idx = 21 + 21;
        }
        else if (pData->vis_type == 4)
        {
            start_idx = 21 + 21;
            end_idx = 21 + 21 + 21;
        }
        else
        {
            assert(pData->vis_type == 5);
            start_idx = 15;
            end_idx = 19;
        }
        for (int i = start_idx; i < end_idx; i++)
        {
            if (pData->resultJoint[3*i+0] < min_s.x) min_s.x = pData->resultJoint[3*i+0];
            if (pData->resultJoint[3*i+0] > max_s.x) max_s.x = pData->resultJoint[3*i+0];
            if (pData->resultJoint[3*i+1] < min_s.y) min_s.y = pData->resultJoint[3*i+1];
            if (pData->resultJoint[3*i+1] > max_s.y) max_s.y = pData->resultJoint[3*i+1];
            if (pData->resultJoint[3*i+2] < min_s.z) min_s.z = pData->resultJoint[3*i+2];
            if (pData->resultJoint[3*i+2] > max_s.z) max_s.z = pData->resultJoint[3*i+2];
        }
        const GLfloat centerx = (min_s.x + max_s.x) / 2;
        const GLfloat centery = (min_s.y + max_s.y) / 2;
        const GLfloat centerz = (min_s.z + max_s.z) / 2;
        glTranslated(-centerx, -centery, -centerz);
    }

    if (pData->bShowBackgroundTexture)
    {
        glUseProgram(0);
        glDisable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_rgbfloatTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, pData->backgroundImage.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBegin(GL_QUADS);
        glColor3f(1.0, 1.0, 1.0);
        const float d = 995;

        glTexCoord2f(0, 0);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(options.K);
        Eigen::Vector3d P(0, 0, 1);
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);  // K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 0);
        P << 1920, 0, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);  // K^{-1} [1920, 0, 1]^T

        glTexCoord2f(1, 1);
        P << 1920, 1080, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);

        glTexCoord2f(0, 1);
        P << 0, 1080, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LIGHTING);
    }

    glUseProgram(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    //Draw the target joint
    if (options.show_joint && pData->targetJoint != NULL)
    {
        glColor3ub(100u, 100u, 100u);
        DrawSkeleton(pData->targetJoint, pData->vis_type, pData->connMat[pData->vis_type]);
    }
    //Draw the predicted joint
    if (options.show_joint && pData->resultJoint != NULL)
    {
        glColor3ub(50u, 50u, 50u);
        DrawSkeleton(pData->resultJoint, pData->vis_type, pData->connMat[pData->vis_type]);
    }

    if (pData->bRenderFloor)  // draw the floor
    {
        glDisable(GL_LIGHTING);
        const int grid_num = 20;
        const int width = 100;

        Eigen::Vector3d normal(pData->ground_normal[0], pData->ground_normal[1], pData->ground_normal[2]);
        normal = normal / normal.norm();
        Eigen::Vector3d tangent1(normal[0], -normal[2], normal[1]), tangent2(normal[1], -normal[0], normal[2]);
        Eigen::Vector3d origin(pData->ground_center[0], pData->ground_center[1], pData->ground_center[2]);
        origin = origin - grid_num * width * (tangent1 + tangent2);
        for (auto y = 0; y < 2 * grid_num + 1; y++)
            for (auto x = 0; x < 2 * grid_num + 1; x++)
            {
                if ((x + y) % 2 == 0)
                    glColor3f(1.0, 1.0, 1.0);
                else
                    glColor3f(0.7, 0.7, 0.7);
                Eigen::Vector3d P = origin + x * width * tangent1 + y * width * tangent2;
                glBegin(GL_QUADS);
                glVertex3f(P[0], P[1], P[2]);
                P = P + width * tangent1;
                glVertex3f(P[0], P[1], P[2]);
                P = P + width * tangent2;
                glVertex3f(P[0], P[1], P[2]);
                P = P - width * tangent1;
                glVertex3f(P[0], P[1], P[2]);
                glEnd();
            }
        glEnable(GL_LIGHTING);
    }

    glPopMatrix();
    glutSwapBuffers();
}

void Renderer::Display()
{
    printf("Starting to display\n");
    glutMainLoop();

}

void Renderer::SpecialKeys(const int key, const int x, const int y)
{
    if(key == GLUT_KEY_UP)
        Renderer::options.xrot -= 2.0;
    else if(key == GLUT_KEY_DOWN)
        Renderer::options.xrot += 2.0;
    else if(key == GLUT_KEY_LEFT)
        Renderer::options.yrot -= 2.0;
    else if(key == GLUT_KEY_RIGHT)
        Renderer::options.yrot += 2.0;
    glutPostRedisplay();
    // Unused arguments (to avoid unused warning)
    (void)key;
    (void)x;
    (void)y;
}

void Renderer::MeshRender()
{
    if (use_color_fbo)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_color);
        glViewport(0, 0, options.width, options.height);
        GLenum FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        // GLfloat white[3] = {1., 1., 1.};
        // glClearBufferfv(GL_COLOR, 0, white);
        if (GL_FRAMEBUFFER_COMPLETE != FBOstatus)
        {
            std::cout << "FrameBuffer Fails." << std::endl;
            exit(0);
        }
        static const GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, draw_buffers);
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, options.width, options.height);
    }

    glClearColor(1., 1., 1., 0.);
    // mesh renderer
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();
    if (options.CameraMode == 0u)
    {
        gluLookAt(0, -options.view_dist * sin(0.0), -options.view_dist * cos(0.0), 0, 0, 0, 0, -1, 0);
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
    }
    else if (options.CameraMode == 2u)
    {
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
        const float centerx = 1920 * options.ortho_scale / 2;
        const float centery = 1080 * options.ortho_scale / 2;
        gluLookAt(centerx, centery, 0, centerx, centery, 1, 0, -1, 0);
    }
    else
    {
        assert(options.CameraMode == 1u);
        glTranslatef(0, 0, options.view_dist);
        glRotatef(options.xrot, 1.0, 0.0, 0.0);
        glRotatef(options.yrot, 0.0, 1.0, 0.0);
        glTranslatef(0, 0, -options.view_dist);
    }

    if (options.CameraMode == 0u)
    {
        cv::Point3d min_s(10000., 10000., 10000.);
        cv::Point3d max_s(-10000., -10000., -10000.);
        assert(pData->resultJoint);
        int start_idx, end_idx;
        if (pData->vis_type <= 2)
        {
            start_idx = 0;
            if (pData->vis_type == 0)  // for hand
                end_idx = 21;
            else if(pData->vis_type == 1)  // for body (SMC order)
                end_idx = 21;
            else if (pData->vis_type == 2) // for hand and body
                end_idx = 62;
        }
        else if(pData->vis_type == 3)
        {
            start_idx = 21;
            end_idx = 21 + 21;
        }
        else if (pData->vis_type == 4)
        {
            start_idx = 21 + 21;
            end_idx = 21 + 21 + 21;
        }
        else
        {
            assert(pData->vis_type == 5);
            start_idx = 15;
            end_idx = 19;
        }
        for (int i = start_idx; i < end_idx; i++)
        {
            if (pData->resultJoint[3*i+0] < min_s.x) min_s.x = pData->resultJoint[3*i+0];
            if (pData->resultJoint[3*i+0] > max_s.x) max_s.x = pData->resultJoint[3*i+0];
            if (pData->resultJoint[3*i+1] < min_s.y) min_s.y = pData->resultJoint[3*i+1];
            if (pData->resultJoint[3*i+1] > max_s.y) max_s.y = pData->resultJoint[3*i+1];
            if (pData->resultJoint[3*i+2] < min_s.z) min_s.z = pData->resultJoint[3*i+2];
            if (pData->resultJoint[3*i+2] > max_s.z) max_s.z = pData->resultJoint[3*i+2];
        }
        const GLfloat centerx = (min_s.x + max_s.x) / 2;
        const GLfloat centery = (min_s.y + max_s.y) / 2;
        const GLfloat centerz = (min_s.z + max_s.z) / 2;
        glTranslated(-centerx, -centery, -centerz);
    }

    if (pData->bShowBackgroundTexture)
    {
        glUseProgram(0);
        glDisable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_rgbfloatTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, pData->backgroundImage.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBegin(GL_QUADS);
        glColor3f(1.0, 1.0, 1.0);
        const float d = 995;

        glTexCoord2f(0, 0);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(options.K);
        Eigen::Vector3d P(0, 0, 1);
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);  // K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 0);
        P << 1920, 0, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);  // K^{-1} [1920, 0, 1]^T

        glTexCoord2f(1, 1);
        P << 1920, 1080, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);

        glTexCoord2f(0, 1);
        P << 0, 1080, 1;
        P = K.inverse() * P;
        P = P / P(2);  // normalize so that z = 1
        glVertex3f(P(0) * d, P(1) * d, P(2) * d);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LIGHTING);
    }

    if (options.show_mesh)
    {
        glEnable(GL_TEXTURE_2D);
        // MVP
        glm::mat4 mvMat,pMat,mvpMat;
        glGetFloatv(GL_MODELVIEW_MATRIX, &mvMat[0][0]);
        glGetFloatv(GL_PROJECTION_MATRIX, &pMat[0][0]);
        mvpMat = pMat * mvMat;

        glUseProgram(g_shaderProgramID[MODE_DRAW_MESH]);
        GLuint MVP_id = glGetUniformLocation(g_shaderProgramID[MODE_DRAW_MESH], "MVP");
        glUniformMatrix4fv(MVP_id, 1, GL_FALSE, &mvpMat[0][0]);

        GLuint MV_id = glGetUniformLocation(g_shaderProgramID[MODE_DRAW_MESH], "MV");
        glUniformMatrix4fv(MV_id, 1, GL_FALSE, &mvMat[0][0]);
        // mesh part
        glUseProgram(g_shaderProgramID[MODE_DRAW_MESH]);
        glLineWidth((GLfloat)0.5);

        // glColor3f(1.0f, 0.0f, 0.0f);
        glBindVertexArray(g_vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer);

        glBufferData(GL_ARRAY_BUFFER, sizeof(cv::Point3d) * (pData->m_meshVertices.size()) * 3 + sizeof(double) * (pData->m_meshVertices.size()),  //3 for vertex and color + 1 for alpha
            NULL, GL_STATIC_DRAW);  
        int offset = sizeof(cv::Point3d) * (pData->m_meshVertices.size());
        void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

        for (auto i = 0u; i< (pData->m_meshVertices).size(); ++i)     //Actual Data copy is done here
        {
            memcpy((char*)ptr + sizeof(cv::Point3d) * i, &(pData->m_meshVertices[i]), sizeof(cv::Point3d));
            memcpy((char*)ptr + sizeof(cv::Point3d) * i + offset, &(pData->m_meshVerticesColor[i]), sizeof(cv::Point3d));
            memcpy((char*)ptr + sizeof(cv::Point3d) * i + offset + offset, &(pData->m_meshVerticesNormal[i]), sizeof(cv::Point3d));
            memcpy((char*)ptr + sizeof(double) * i + offset + offset + offset, &(pData->m_meshVerticesAlpha[i]), sizeof(double));
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);     // Tell OpenGL that we're done with the pointer

        GLuint vertexPosition_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_MESH], "vertex_pos");
        glVertexAttribPointer(vertexPosition_id,            // Attribute 0 or vertexPosition_id
            3,            // size
            GL_DOUBLE,     // Type: Floating-point data
            GL_FALSE,     // Not normalized
                          // (floating-point data never is)
            0,            // stride
            0);        // Offset zero (NULL pointer)
        glEnableVertexAttribArray(vertexPosition_id);

        GLuint vertexColor_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_MESH], "vertex_color");
        glVertexAttribPointer(vertexColor_id,            // Attribute 1 or vertexColor_id
            3,            // size
            GL_DOUBLE,     // Type: Floating-point data
            GL_FALSE,     // Not normalized
                          // (floating-point data never is)
            0,            // stride
            (void*)offset);        // Offset in the buffer 
        glEnableVertexAttribArray(vertexColor_id);

        //Vertex Normal
        // glBindBuffer(GL_ARRAY_BUFFER, g_normalBuffer);      //Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
        // glBufferData(GL_ARRAY_BUFFER, pData->m_meshVerticesNormal.size() * sizeof(cv::Point3d), &(pData->m_meshVerticesNormal[0]), GL_STATIC_DRAW); //allocate and copy togetherd
        GLuint vertexNormal_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_MESH], "vertex_normal");
        glVertexAttribPointer(vertexNormal_id,            // Attribute 1 or vertexColor_id
            3,            // size
            GL_DOUBLE,     // Type: Floating-point data
            GL_FALSE,     // Not normalized
                            // (floating-point data never is)
            0,            // stride
            (void*)(offset + offset));
            // (void*)0);        // Offset in the buffer 
        glEnableVertexAttribArray(vertexNormal_id);

        GLuint vertexAlpha_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_MESH], "vertex_alpha");
        glVertexAttribPointer(vertexAlpha_id,            // Attribute 1 or vertexColor_id
            1,            // size
            GL_DOUBLE,     // Type: Floating-point data
            GL_FALSE,     // Not normalized
                            // (floating-point data never is)
            0,            // stride
            (void*)(offset + offset + offset));
            // (void*)0);        // Offset in the buffer 
        glEnableVertexAttribArray(vertexAlpha_id);

        //Face indexing
        if (pData->m_meshIndices.size()>0)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);       //Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, pData->m_meshIndices.size() * sizeof(unsigned int), &(pData->m_meshIndices[0]), GL_STATIC_DRAW);  //allocate and copy together
        }

        // if (options.meshSolid) glPolygonMode(GL_FRONT, GL_FILL);
        if (options.meshSolid) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        if (pData->m_meshIndices.size() > 0)
        {
            glDrawElements(         //Indexed
                GL_TRIANGLES,      // mode
                pData->m_meshIndices.size(),    // count
                GL_UNSIGNED_INT,   // type
                (void*)0           // element array buffer offset
            );
        }
        else glDrawArrays(GL_POINTS, 0, pData->m_meshVertices.size());   //Non indexing version
        glDisable(GL_TEXTURE_2D);
    }

    glUseProgram(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    //Draw the target joint
    if (options.show_joint && pData->targetJoint != NULL)
    {
        glColor3ub(0u, 255u, 0u);
        DrawSkeleton(pData->targetJoint, pData->vis_type, pData->connMat[pData->vis_type]);
    }
    //Draw the predicted joint
    if (options.show_joint && pData->resultJoint != NULL)
    {
        glColor3ub(50u, 50u, 50u);
        DrawSkeleton(pData->resultJoint, pData->vis_type, pData->connMat[pData->vis_type]);
    }

    if (pData->bRenderFloor)  // draw the floor
    {
        glDisable(GL_LIGHTING);
        const int grid_num = 20;
        const int width = 100;

        Eigen::Vector3d normal(pData->ground_normal[0], pData->ground_normal[1], pData->ground_normal[2]);
        normal = normal / normal.norm();
        Eigen::Vector3d tangent1(normal[0], -normal[2], normal[1]), tangent2(normal[1], -normal[0], normal[2]);
        Eigen::Vector3d origin(pData->ground_center[0], pData->ground_center[1], pData->ground_center[2]);
        origin = origin - grid_num * width * (tangent1 + tangent2);
        for (auto y = 0; y < 2 * grid_num + 1; y++)
            for (auto x = 0; x < 2 * grid_num + 1; x++)
            {
                if ((x + y) % 2 == 0)
                    glColor3f(1.0, 1.0, 1.0);
                else
                    glColor3f(0.7, 0.7, 0.7);
                Eigen::Vector3d P = origin + x * width * tangent1 + y * width * tangent2;
                glBegin(GL_QUADS);
                glVertex3f(P[0], P[1], P[2]);
                P = P + width * tangent1;
                glVertex3f(P[0], P[1], P[2]);
                P = P + width * tangent2;
                glVertex3f(P[0], P[1], P[2]);
                P = P - width * tangent1;
                glVertex3f(P[0], P[1], P[2]);
                glEnd();
            }
        glEnable(GL_LIGHTING);
    }

    glPopMatrix();
    glutSwapBuffers();
}

void Renderer::DrawSkeleton(double* joint, uint vis_type, std::vector<int> connMat)
{
    int start_idx = 0, end_idx;
    float rad, cone_rad;
    if (vis_type == 0)  // for hand
    {
        end_idx = 21;
        rad = 0.2f;
        cone_rad = 0.1f;
    }
    else if(vis_type == 1)  // for body (SMC order)
    {
        end_idx = 20;
        rad = 2.0f;
        cone_rad = 2.0f;
    }
    else if (vis_type == 2) // for hand and body
    {
        end_idx = 62;
        rad = 1.0f;
        cone_rad = 0.3f;
    }
    else if(vis_type == 3)
    {
        start_idx = 20;
        end_idx = 20 + 21;
        rad = 0.2f;
        cone_rad = 0.1;
    }
    else
    {
        assert(vis_type == 4);
        start_idx = 20 + 21;
        end_idx = 20 + 21 + 21;
        rad = 0.2f;
        cone_rad = 0.1;
    }

    if (vis_type == 2)
    {
        for (int i = start_idx; i < end_idx; i++)
        {
            if (i < 21) rad = 2.0f;
            else rad = 1.0f;
            glPushMatrix();
            glTranslated(joint[3*i], joint[3*i+1], joint[3*i+2]);
            glutSolidSphere(rad, 100, 100);
            glPopMatrix();
        }

        for (uint i = 0; i < connMat.size() / 2; i++)
        {
            int j = connMat[2 * i];
            int k = connMat[2 * i + 1];
            if (j < 21)  cone_rad = 2.0f;
            else cone_rad = 1.0f;
            GLfloat x0 = joint[3 * j], y0 = joint[3 * j + 1], z0 = joint[3 * j + 2];
            GLfloat x1 = joint[3 * k] - x0, y1 = joint[3 * k + 1] - y0, z1 = joint[3 * k + 2] - z0;
            GLfloat length = sqrt(x1*x1 + y1*y1 + z1*z1);
            GLfloat theta = acos(z1/length) * 180 / PI;
            GLfloat phi = atan2(y1, x1) * 180 / PI;
            glPushMatrix();
            glTranslatef(x0, y0, z0);
            glRotatef(phi, 0, 0, 1);
            glRotatef(theta, 0, 1, 0);
            glutSolidCone(cone_rad, length, 100, 100);
            glPopMatrix();
        }
    }
    else
    {
        for (int i = start_idx; i < end_idx; i++)
        {
            glPushMatrix();
            glTranslated(joint[3*i], joint[3*i+1], joint[3*i+2]);
            glutSolidSphere(rad, 100, 100);
            glPopMatrix();
        }

        for (uint i = 0; i < connMat.size() / 2; i++)
        {
            int j = connMat[2 * i];
            int k = connMat[2 * i + 1];
            GLfloat x0 = joint[3 * j], y0 = joint[3 * j + 1], z0 = joint[3 * j + 2];
            GLfloat x1 = joint[3 * k] - x0, y1 = joint[3 * k + 1] - y0, z1 = joint[3 * k + 2] - z0;
            GLfloat length = sqrt(x1*x1 + y1*y1 + z1*z1);
            GLfloat theta = acos(z1/length) * 180 / PI;
            GLfloat phi = atan2(y1, x1) * 180 / PI;
            glPushMatrix();
            glTranslatef(x0, y0, z0);
            glRotatef(phi, 0, 0, 1);
            glRotatef(theta, 0, 1, 0);
            glutSolidCone(cone_rad, length, 100, 100);
            glPopMatrix();
        }
    }
}

// void Renderer::IdleSaveImage()
// {
//     BYTE* pixels = new BYTE[3 * options.width * options.height];
//     glReadPixels(0, 0, options.width, options.height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
//     FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, options.width, options.height, 3 * options.width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
//     FreeImage_Save(FIF_JPEG, image, "/home/donglaix/temp.jpg", 0);
//     delete[] pixels;
// }

void Renderer::RenderAndRead()
{
    // if (use_color_fbo) glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_color);
    glutMainLoopEvent();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // glReadPixels(0, 0, options.width, options.height, GL_RGB, GL_UNSIGNED_BYTE, pData->read_buffer);
    glReadPixels(0, 0, options.width, options.height, GL_RGBA, GL_UNSIGNED_BYTE, pData->read_buffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutPostRedisplay();
    g_drawMode = MODE_DRAW_DEFUALT;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

void Renderer::CameraMode(uint position, int width, int height, double* calibK)
{
    options.width = width; options.height = height;
    if(calibK != NULL) options.K = calibK;
    glutReshapeWindow(width, height);
    options.CameraMode = 1u;
    if (position == 0)
    {
        // look from the front
        options.xrot = 0;
        options.yrot = 0;
    }
    else if (position == 1)
    {
        // look down
        options.xrot = 90;
        options.yrot = 0;
    }
    else if (position == 2)
    {
        // look from the left
        options.xrot = 0;
        options.yrot = 90;
    }
}

void Renderer::NormalMode(uint position, int width, int height)
{
    if (position == 0)
    {
        // look from the front
        options.xrot = 0;
        options.yrot = 0;
    }
    else if (position == 1)
    {
        // look down
        options.xrot = 90;
        options.yrot = 0;
    }
    else if (position == 2)
    {
        // look from the right
        options.xrot = 0;
        options.yrot = 90;
    }
    else if (position == 3)
    {
        // look from the bottom
        options.xrot = -90;
        options.yrot = 0;
    }
    else if (position == 4)
    {
        // look from the left
        options.xrot = 0;
        options.yrot = -90;
    }
    else
    {
        printf("Invalid position type(0, 1, 2, 3, 4).\n");
        exit(1);
    }
    options.width = width; options.height = height;
    glutReshapeWindow(width, height);
    options.CameraMode = 0u;
}

void Renderer::OrthoMode(float scale, uint position)
{
    options.width = 1920; options.height = 1080;
    glutReshapeWindow(options.width, options.height);
    options.ortho_scale = scale;
    options.CameraMode = 2u;
    if (position == 0)
    {
        // look from the front
        options.xrot = 0;
        options.yrot = 0;
    }
    else if (position == 1)
    {
        // look down
        options.xrot = 90;
        options.yrot = 0;
    }
    else if (position == 2)
    {
        // look from the left
        options.xrot = 0;
        options.yrot = 90;
    }
}

void Renderer::RenderDepthMap(VisualizedData& g_visData)
{
    pData = &g_visData;
    glutDisplayFunc(DepthMapRenderer);
}

void Renderer::DepthMapRenderer()
{
    g_drawMode = MODE_DRAW_DEPTH_RENDER;
    glUseProgram(g_shaderProgramID[g_drawMode]);

    GLuint MVP_id = glGetUniformLocation(g_shaderProgramID[g_drawMode], "MVP");
    // Get the current MVP matrix
    glm::mat4 mvMat,pMat,mvpMat;
    glGetFloatv(GL_MODELVIEW_MATRIX, &mvMat[0][0]);
    glGetFloatv(GL_PROJECTION_MATRIX, &pMat[0][0]);
    mvpMat = pMat * mvMat;
    glUniformMatrix4fv(MVP_id, 1, GL_FALSE, &mvpMat[0][0]);

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer);

    glBufferData(GL_ARRAY_BUFFER, sizeof(cv::Point3d) * pData->m_meshVertices.size(),  
        &(pData->m_meshVertices[0]), GL_STATIC_DRAW);

    GLuint vertexPosition_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_DEPTH_RENDER], "vertex_pos");
    glVertexAttribPointer(vertexPosition_id,            // Attribute 0 or vertexPosition_id
        3,            // size
        GL_DOUBLE,     // Type: Floating-point data
        GL_FALSE,     // Not normalized
                      // (floating-point data never is)
        0,            // stride
        0);        // Offset zero (NULL pointer)
    glEnableVertexAttribArray(vertexPosition_id);

    glEnable(GL_TEXTURE_2D);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_depth);
    glViewport(0, 0, 1920, 1080);
    GLenum FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (GL_FRAMEBUFFER_COMPLETE != FBOstatus)
    {
        std::cout << "FrameBuffer Fails." << std::endl;
        exit(0);
    }
    float one = 1.0;
    glClearBufferfv(GL_DEPTH, 0, &one);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawBuffer(GL_NONE);

    if (pData->m_meshIndices.size() > 0)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);       //Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, pData->m_meshIndices.size() * sizeof(unsigned int), &(pData->m_meshIndices[0]), GL_STATIC_DRAW);  //allocate and copy together
    }

    if (options.meshSolid) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    if (pData->m_meshIndices.size() > 0)
    {
        glDrawElements(         //Indexed
            GL_TRIANGLES,      // mode
            pData->m_meshIndices.size(),    // count
            GL_UNSIGNED_INT,   // type
            (void*)0           // element array buffer offset
        );
    }
    else glDrawArrays(GL_POINTS, 0, pData->m_meshVertices.size());   //Non indexing version
    glutSwapBuffers();
}

void Renderer::RenderAndReadDepthMap()
{
    glutMainLoopEvent();
    if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
        std::cout << "FrameBuffer Fails." << std::endl;
        exit(0);
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadPixels(0, 0, 1920, 1080, GL_DEPTH_COMPONENT, GL_FLOAT, pData->read_depth_buffer);
    const double nearFar_interval = options.zmax - options.zmin;
    const double nearFar_numerator = - options.zmax * options.zmin / nearFar_interval;
    for (int i = 0; i < 1920 * 1080; i++) pData->read_depth_buffer[i] = nearFar_numerator / (pData->read_depth_buffer[i] - options.zmax / nearFar_interval);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutPostRedisplay();
    g_drawMode = MODE_DRAW_DEFUALT;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

void Renderer::RenderProjection(VisualizedData& g_visData)
{
    pData = &g_visData;
    glutDisplayFunc(ProjectionRenderer);
}

void Renderer::ProjectionRenderer()
{
    assert(options.CameraMode == 1u);
    g_drawMode = MODE_DRAW_PROJECTION;
    glUseProgram(g_shaderProgramID[g_drawMode]);

    GLuint MVP_id = glGetUniformLocation(g_shaderProgramID[g_drawMode], "MVP");
    // Get the current MVP matrix
    glm::mat4 mvMat,pMat,mvpMat;
    glGetFloatv(GL_MODELVIEW_MATRIX, &mvMat[0][0]);
    glGetFloatv(GL_PROJECTION_MATRIX, &pMat[0][0]);
    mvpMat = pMat * mvMat;
    glUniformMatrix4fv(MVP_id, 1, GL_FALSE, &mvpMat[0][0]);

    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo_rgbfloat);
    glViewport(0, 0, 1920, 1080);
    GLenum FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (GL_FRAMEBUFFER_COMPLETE != FBOstatus)
    {
        std::cout << "FrameBuffer Fails." << std::endl;
        exit(0);
    }
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    GLfloat white[3] = {-1., -1., -1.};  // test value
    glClearBufferfv(GL_COLOR, 0, white);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer);

    glBufferData(GL_ARRAY_BUFFER, sizeof(cv::Point3d) * (pData->m_meshVertices.size()) + sizeof(cv::Point3d) * (pData->m_meshVerticesColor.size()),
        NULL, GL_STATIC_DRAW);  
    int offset = sizeof(cv::Point3d) * (pData->m_meshVertices.size());
    void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    for (auto i = 0u; i< (pData->m_meshVertices).size(); ++i)     //Actual Data copy is done here
    {
        memcpy((char*)ptr + sizeof(cv::Point3d) * i, &(pData->m_meshVertices[i]), sizeof(cv::Point3d));
        memcpy((char*)ptr + sizeof(cv::Point3d) * i + offset, &(pData->m_meshVerticesColor[i]), sizeof(cv::Point3d));   // the 2D projection is suppose to be store in m_meshVerticesUV
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);     // Tell OpenGL that we're done with the pointer

    GLuint vertexPosition_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_PROJECTION], "vertex_pos");

    glVertexAttribPointer(vertexPosition_id,            // Attribute 0 or vertexPosition_id
        3,            // size
        GL_DOUBLE,     // Type: Floating-point data
        GL_FALSE,     // Not normalized
                      // (floating-point data never is)
        0,            // stride
        0);        // Offset zero (NULL pointer)
    glEnableVertexAttribArray(vertexPosition_id);

    GLuint vertexColor_id = glGetAttribLocation(g_shaderProgramID[MODE_DRAW_PROJECTION], "coord2D_in");

    glVertexAttribPointer(vertexColor_id,            // Attribute 1 or vertexColor_id
        3,            // size
        GL_DOUBLE,     // Type: Floating-point data
        GL_FALSE,     // Not normalized
                      // (floating-point data never is)
        0,            // stride
        (void*)offset);        // Offset in the buffer 
    glEnableVertexAttribArray(vertexColor_id);

    //Face indexing
    if (pData->m_meshIndices.size()>0)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBuffer);       //Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, pData->m_meshIndices.size() * sizeof(unsigned int), &(pData->m_meshIndices[0]), GL_STATIC_DRAW);  //allocate and copy together
    }
    glPolygonMode(GL_FRONT, GL_FILL);
    if (pData->m_meshIndices.size() > 0)
    {
        glDrawElements(         //Indexed
            GL_TRIANGLES,      // mode
            pData->m_meshIndices.size(),    // count
            GL_UNSIGNED_INT,   // type
            (void*)0           // element array buffer offset
        );
    }
    else glDrawArrays(GL_POINTS, 0, pData->m_meshVertices.size());   //Non indexing version
    glEnable(GL_LIGHTING);
    glFlush();
    glutSwapBuffers();
}

void Renderer::RenderAndReadProjection()
{
    glutMainLoopEvent();
    if (GL_FRAMEBUFFER_COMPLETE != glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
        std::cout << "FrameBuffer Fails." << std::endl;
        exit(0);
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadPixels(0, 0, 1920, 1080, GL_RGB, GL_FLOAT, pData->read_rgbfloat_buffer);
    glutPostRedisplay();
    g_drawMode = MODE_DRAW_DEFUALT;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}
