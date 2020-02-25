#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc

#include "tiny_obj_loader.h"
#include <iostream>

struct ProgramOptions;
struct Mesh;
void get_options(const int argc, char* const* argv, ProgramOptions &options);
void print_options(const ProgramOptions &options);

struct ProgramOptions
{
    std::string obj_file;
    std::string mtl_basedir;
    std::array<int, 2> image_size;
    std::array<float, 2> focal_length;
    std::array<float, 2> principal_point;
    std::array<float, 12> projection_matrix;
}; //endstruct

struct Mesh
{
    std::string obj_file;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
}; // nedstruct Mesh

void get_options(const int argc, char* const* argv, ProgramOptions &options)
{
    if (argc < 20)
    {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <obj_file> <mtl_basedir> <image_size (w, h)>"
                     " <focal_length (fx, fy)> <principal_point (cx, cy)>"
                     " <projection_matrix (3x4)>"
                  << std::endl;
        exit(1);
    }
    options.obj_file = argv[1];
    options.mtl_basedir = argv[2];
    options.image_size[0] = std::atoi(argv[3]);
    options.image_size[1] = std::atoi(argv[4]);
    options.focal_length[0] = std::atof(argv[5]);
    options.focal_length[1] = std::atof(argv[6]);
    options.principal_point[0] = std::atof(argv[7]);
    options.principal_point[1] = std::atof(argv[8]);
    for (size_t i = 0; i < 12; i++)
    {
        options.projection_matrix[i] = std::atof(argv[9 + i]);
    }
}

void print_options(const ProgramOptions &options)
{
    std::cout << "Obj file: " << options.obj_file << std::endl;
    std::cout << "mtl_basedir: " << options.mtl_basedir << std::endl;
    std::cout << "image size: "
              << options.image_size[0] << ", " << options.image_size[1]
              << std::endl;
    std::cout << "focal length: "
              << options.focal_length[0] << ", " << options.focal_length[1]
              << std::endl;
    std::cout << "principal point: "
              << options.principal_point[0] << ", "
              << options.principal_point[1]
              << std::endl;
    std::cout << "projection matrix: " << std::endl;
    for (size_t i = 0; i < 3; i++)
    {
        const size_t row_offset = i * 3;
        for (size_t j = 0; j < 4; j++)
        {
            std::cout << options.projection_matrix[row_offset + j] << ", ";
        }
        std::cout << std::endl;
    }
}

bool loadObj(const std::string &obj_file, const std::string &mtl_basedir, Mesh &mesh)
{
    std::string warn;
    std::string err;

    mesh.obj_file = obj_file;
    bool ret = tinyobj::LoadObj(
        &mesh.attrib, &mesh.shapes, &mesh.materials,
        &warn, &err, obj_file.c_str(), mtl_basedir.c_str()
    );

    if (!warn.empty())
    {
      std::cout << warn << std::endl;
    }

    if (!err.empty())
    {
      std::cerr << err << std::endl;
      return false;
    }

    if (!ret)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void print_mesh_stats(const Mesh &mesh)
{
    std::cout << "obj file: " << mesh.obj_file << std::endl;
    std::cout << "vertices: " << mesh.attrib.vertices.size() << std::endl;
    std::cout << "shapes: " << mesh.shapes.size() << std::endl;
    for (const auto &shape: mesh.shapes)
    {
        std::cout << "faces: " << shape.mesh.num_face_vertices.size()
                  << std::endl;
    }
}

int main(int argc, char **argv)
{
    ProgramOptions options;
    get_options(argc, argv, options);
    print_options(options);

    Mesh mesh;
    auto mesh_status = loadObj(options.obj_file, options.mtl_basedir, mesh);
    if (!mesh_status)
    {
        std::cerr << "Could not read mesh" << std::endl;
        exit(1);
    }
    print_mesh_stats(mesh);


}
