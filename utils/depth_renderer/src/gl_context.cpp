/**
 * @file gl_context.cpp
 * @author Rakesh Shrestha, rakeshs@sfu.ca
 */

#include <p2mpp_depth_renderer/gl_context.h>
#include <p2mpp_depth_renderer/offscreen_rendering.h>

#include <iostream>
using namespace std;
using namespace p2mpp_depth_renderer;

void GlContext::openWindow()
{
    const int WIDTH = 640;
    const int HEIGHT = 480;
    const char *TITLE = "Please ignore me";

    int monitorWidth, monitorHeight;
    GLenum error;

    // we need to intialize GLFW before we create a GLFW window
    if(!glfwInit())
    {
        cerr << "openWindow() could not initialize GLFW" << endl;
        exit(1);
    }

    // explicitly set our OpenGL context to something new enough to support the kind of framebuffer functions we need
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 0);							// allow deprecated functionality
    glfwWindowHint(GLFW_REFRESH_RATE, 60);
    glfwWindowHint(GLFW_RESIZABLE, 0);
    glfwWindowHint(GLFW_DECORATED, 1);
    glfwWindowHint(GLFW_FOCUSED, 1);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
    glfwWindowHint(GLFW_VISIBLE, 0);

    // use the current desktop mode to decide on a suitable resolution
    const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    monitorWidth = mode -> width;
    monitorHeight = mode -> height;

    // create our OpenGL window using GLFW
    window = glfwCreateWindow(WIDTH, HEIGHT,					// specify width and height
                              TITLE,							// title of window
                              NULL,								// always windowed mode
                              NULL);							// not sharing resources across monitors
    if(!window)
    {
        cerr << "openWindow() could not create an OpenGL window---do your GL context/profile hints make sense?" << endl;
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowPos(window,									// center the window
                     (monitorWidth / 2.0) - (WIDTH / 2),
                     (monitorHeight / 2.0) - (HEIGHT / 2));
    glfwSwapInterval(1);

    // configure our viewing area
    glViewport(0, 0, WIDTH, HEIGHT);

    // enable our extensions handler
    glewExperimental = true;		// GLEW bug: glewInit() doesn't get all extensions, so we have it explicitly search for everything it can find
    error = glewInit();
    if(error != GLEW_OK)
    {
        cerr << glewGetErrorString(error) << endl;
        exit(1);
    }

    // clear the OpenGL error code that results from initializing GLEW
    glGetError();

    // save our window size for later on
    windowSize = vec2(WIDTH, HEIGHT);

    // print our OpenGL version info
    cout << "-- GL version:   " << (char*)glGetString(GL_VERSION) << endl;
    cout << "-- GL vendor:    " << (char*)glGetString(GL_VENDOR) << endl;
    cout << "-- GL renderer:  " << (char*)glGetString(GL_RENDERER) << endl;
    cout << "-- GLSL version: " << (char*)glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

    // turn on OpenGL debugging
    // initGLDebugger();
}

void GlContext::prepareOpenGL()
{
    // turn on depth testing and disable blending
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // white screen in background
    glClearColor(1.0, 1.0, 1.0, 1.0);

    // make sure nothing funky goes on---render both front and back of all geometric primitives
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
}

void GlContext::prepareFramebufferRendering()
{
    glGenFramebuffers(1, &sceneFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);

    // set up the texture we'll be rendering colour into
    glGenTextures(1, &sceneColorTexture);
    glBindTexture(GL_TEXTURE_2D, sceneColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowSize.x, windowSize.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // set up the texture we'll be rendering depth into
    glGenTextures(1, &sceneDepthTexture);
    glBindTexture(GL_TEXTURE_2D, sceneDepthTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize.x, windowSize.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    // make sure we've got a depth buffer, too, since we're rendering 3D objects into this framebuffer
    glGenRenderbuffers(1, &sceneDepthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, sceneDepthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowSize.x, windowSize.y);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sceneDepthBuffer);

    // attach our textures to our framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColorTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sceneDepthTexture, 0);
    //GLenum drawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
    //glDrawBuffers(2, drawBuffers);
    // for some reason this above line throws a GL callback debugging error in GL 3.0 contexts...

    // make sure everything was set up properly
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        cerr << "prepareFramebufferRendering(): incomplete FBO: " << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
        exit(1);
    }
}

void GlContext::initialize()
{
    openWindow();
    prepareOpenGL();
    prepareFramebufferRendering();
    renderer_initialized = true;
}

std::vector<float> GlContext::render(
        std::vector<Eigen::Vector3f> vertices,
        std::vector<Eigen::Matrix<GLushort, -1, 3>> faces,
        std::vector<Eigen::Vector3f> cams,
        std::array<GLsizei, 2> image_size,
        float scale, bool inverse
)
{
    if (!renderer_initialized)
    {
        initialize();
    }
    return offscreen_rendering::render(
        vertices, faces, cams, image_size, scale, inverse
    );

}

std::vector<float> GlContext::render_threaded(
        std::vector<Eigen::Vector3f> vertices,
        std::vector<Eigen::Matrix<GLushort, -1, 3>> faces,
        std::vector<Eigen::Vector3f> cams,
        std::array<GLsizei, 2> image_size,
        float scale, bool inverse
)
{
    if (!run_thread_initialized)
    {
        std::cerr << "Run thread not started before render_threaded"
                  << std::endl;
        return std::vector<float>();
    }

    while (!request_new_frame);
    this->vertices = vertices;
    this->faces = faces;
    this->cams = cams;
    this->image_size = image_size;
    this->scale = scale;
    this->inverse = inverse;

    request_new_frame = true;

    while (request_new_frame);
    return depth_buffer;
}

void GlContext::run()
{
    initialize();
    run_thread_initialized = true;
    while (true)
    {
        if (request_new_frame)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (vertices.size() && faces.size())
            {
                depth_buffer = offscreen_rendering::render(
                    vertices, faces, cams, image_size
                );
            }
            request_new_frame = false;
        }
    }
}

