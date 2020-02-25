#ifndef P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H
#define P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H

// #include "utils.h"
// #include "config.h"
#include <Eigen/Geometry>

// glew needs to be before OpenGL
#include "GL/glew.h"
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cmath>
#include <iostream>
#include <iomanip>
//#include <commctrl.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <string>

#include <Eigen/Core>
/**
 * offscreen rendering for getting depth image from 3D model
 */
namespace p2mpp_depth_renderer
{
namespace offscreen_rendering
{

typedef std::array<GLsizei, 2> image_size_t;

constexpr float toFOV(float focal_length, float image_size)
{
    return 2 * std::atan2(image_size, (2 * focal_length));
}

constexpr float toDegrees(float radians)
{
    return radians * 180.0f / M_PI;
}

class ProjectionParameters
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionParameters(Eigen::Matrix4f extrinsics, float focal_length,
                         image_size_t _image_size, float _z_near, float _z_far,
                         float _scale=1.0, bool _inverse=false)
        : T_cam_world(extrinsics),
          fovy(toDegrees(toFOV(focal_length, _image_size[1]))),
          z_near(_z_near), z_far(_z_far),
          image_size(_image_size), inverse(_inverse)
    { }
    ~ProjectionParameters() = default;
    // world to cam transformation
    Eigen::Matrix<float, 4, 4, Eigen::ColMajor> T_cam_world;
    float fovy;
    image_size_t image_size;
    float z_near;
    float z_far;
    float scale;
    bool inverse;
}; // endclass ProjectionParameters

/**
 *
 * @param vertices vector of vertices for each segment (the vertices are stored as float vectors)
 * @param faces vector of indices that make faces from the vertices
 * @param cams
 * @param inverse
 */
void mGLRender(const Eigen::Matrix<float, -1, 3> &vertices,
               const Eigen::Matrix<GLushort, -1, 3> &faces,
               const ProjectionParameters &projection_parameters);

/**
 *
 * @param vertices vector of vertices for each segment (the vertices are stored a float vectors for each segment)
 * @param faces vector of indices that make faces from the vertices
 * @param cams
 * @param inverse
 * @return
 */
std::vector<float> render(const Eigen::Matrix<float, -1, 3> &vertices,
                          const Eigen::Matrix<GLushort, -1, 3> &faces,
                          const ProjectionParameters &projection_parameters);

} // endnamespace offscren_rendering
} // endnamespace p2mpp_depth_renderer


#endif // P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H
