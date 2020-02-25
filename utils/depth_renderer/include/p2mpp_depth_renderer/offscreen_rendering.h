#ifndef P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H
#define P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H

// #include "utils.h"
// #include "config.h"
#include <Eigen/Geometry>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <iostream>
#include <iomanip>
//#include <commctrl.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <string>

/**
 * offscreen rendering for getting depth image from 3D model
 */
namespace p2mpp_depth_renderer
{
namespace offscreen_rendering
{

/**
 *
 * @param vertices vector of vertices for each segment (the vertices are stored as float vectors)
 * @param faces vector of indices that make faces from the vertices
 * @param cams
 * @param inverse
 */
void mGLRender(std::vector<Eigen::Vector3f> vertices,
               std::vector< Eigen::Matrix<GLushort,-1,3> > faces,
               std::vector<Eigen::Vector3f> cams,
               float scale = 1.0, bool inverse = false);

/**
 *
 * @param vertices vector of vertices for each segment (the vertices are stored a float vectors for each segment)
 * @param faces vector of indices that make faces from the vertices
 * @param cams
 * @param inverse
 * @return
 */
std::vector<float> render(std::vector<Eigen::Vector3f> vertices,
                          std::vector<Eigen::Matrix<GLushort, -1, 3>> faces,
                          std::vector<Eigen::Vector3f> cams,
                          std::array<GLsizei, 2> image_size,
                          float scale = 1.0, bool inverse = false);

} // endnamespace offscren_rendering
} // endnamespace p2mpp_depth_renderer


#endif // P2MPP_DEPTH_RENDERER_OFFSCREEN_RENDERING_H
