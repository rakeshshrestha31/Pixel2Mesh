/**
 * @file mesh.cpp
 * @author Rakesh Shrestha, rakeshs@sfu.ca
 */

#ifndef P2MPP_DEPTH_RENDERER_MESH_H
#define P2MPP_DEPTH_RENDERER_MESH_H

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc

#include <string>
#include <Eigen/Core>
#include "GL/glew.h"
#include <GL/gl.h>

#include "tiny_obj_loader.h"

namespace p2mpp_depth_renderer
{

class Mesh
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool loadObj(const std::string &obj_file, const std::string &mtl_basedir);
    void printMeshStats();
    void updateVerticesProperties();
    void updateFacesProperties();
    void writePlyFile(const std::string &ply_filename);
    Eigen::Matrix<float, -1, 3> getVertices() { return vertices_; }
    Eigen::Matrix<GLushort, -1, 3> getFaces() { return faces_; }

protected:
    std::string obj_file_;
    std::string mtl_basedir_;
    tinyobj::attrib_t attrib_;
    std::vector<tinyobj::shape_t> shapes_;
    std::vector<tinyobj::material_t> materials_;
    Eigen::Matrix<float, -1, 3> vertices_;
    Eigen::Matrix<GLushort, -1, 3> faces_;
}; // endclass Mesh
} // endnamespace p2mpp_depth_renderer
#endif // P2MPP_DEPTH_RENDERER_MESH_H
