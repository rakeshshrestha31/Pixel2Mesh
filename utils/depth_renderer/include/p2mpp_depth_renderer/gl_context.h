/**
 * @file gl_context.h
 * @author Rakesh Shrestha, rakeshs@sfu.ca
 */

#ifndef P2MPP_DEPTH_RENDERER_GL_CONTEXT_H
#define P2MPP_DEPTH_RENDERER_GL_CONTEXT_H

#include "GL/glew.h"
#include <GLFW/glfw3.h>                             // context, window, and input handling

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"								// OpenGL Mathematics library to handle our vector and matrix math
#include "glm/gtc/matrix_transform.hpp"				// for lookAt() and perspective(), since we shouldn't use GLU for these (GLU is deprecated)
#include "glm/gtc/type_ptr.hpp"						// for value_ptr() when accessing matrices as floating-point arrays
#include "glm/gtx/rotate_vector.hpp"				// used to compute camera viewing angles when rotating around the center of the world

#include <Eigen/Core>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>

namespace p2mpp_depth_renderer
{

// forward declare
namespace offscreen_rendering
{
    class ProjectionParameters;
}

using namespace glm;
using ProjectionParameters = offscreen_rendering::ProjectionParameters;

class GlContext
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GlContext()
        : request_new_frame(false), run_thread_initialized(false),
          renderer_initialized(false) {}
    ~GlContext();

    // - - - OpenGL configuration and other low-level stuff is all below - - - //
    void openWindow();
    void prepareOpenGL();
    void prepareFramebufferRendering();
    void initialize();

    /**
     *
     * @brief wrapper around the offscreen render in new opengl context
     * @param vertices
     * @param faces
     * @param cams
     * @param total
     * @param inverse
     * @param maxD
     * @return
     */
    std::vector<float> render(const Eigen::Matrix<float, -1, 3> &vertices,
                              const Eigen::Matrix<GLushort, -1, 3> &faces,
                              const ProjectionParameters &projection_parameters);

    std::vector<float> render_threaded(
            const Eigen::Matrix<float, -1, 3> &vertices,
            const Eigen::Matrix<GLushort, -1, 3> &faces,
            const ProjectionParameters &projection_parameters
    );
    void run();

protected:
    // GLFW window and characteristics
    GLFWwindow *window;
    vec2 windowSize;

// framebuffer rendering objects and textures
    GLuint sceneFBO;
    GLuint sceneColorTexture;
    GLuint sceneDepthTexture;
    GLuint sceneDepthBuffer;

    Eigen::Matrix<float, -1, 3> vertices;
    Eigen::Matrix<GLushort, -1, 3> faces;
    std::unique_ptr<ProjectionParameters> projection_parameters;

    std::vector<float> depth_buffer;
    std::atomic_bool run_thread_initialized;
    std::atomic_bool renderer_initialized;
    std::atomic_bool request_new_frame;
    std::mutex mutex_;
};

} // endnamespace p2mpp_depth_renderer
#endif // P2MPP_DEPTH_RENDERER_GL_CONTEXT_H
