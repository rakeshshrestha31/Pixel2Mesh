#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc

#include <p2mpp_depth_renderer/offscreen_rendering.h>
#include <p2mpp_depth_renderer/gl_context.h>
#include <p2mpp_depth_renderer/mesh.h>

#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>


constexpr float MESH_SCALE_FACTOR = 0.57;

struct ProgramOptions;

void get_options(const int argc, char* const* argv, ProgramOptions &options);
void print_options(const ProgramOptions &options);

struct ProgramOptions
{
    std::string obj_file;
    std::string mtl_basedir;
    std::array<int, 2> image_size;
    float focal_length;
    float z_near;
    float z_far;
    std::array<float, 2> principal_point;
    Eigen::Matrix<float, 4, 4, Eigen::ColMajor> extrinsics;
}; //endstruct

void get_options(const int argc, char* const* argv, ProgramOptions &options)
{
    if (argc < 21)
    {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <obj_file> <mtl_basedir> <image_size (w, h)>"
                     " <focal_length fy> <z_near> <z_far> <principal_point (cx, cy)>"
                     " <projection_matrix (3x4)>"
                  << std::endl;
        exit(1);
    }
    int i = 1;
    options.obj_file = argv[i++];
    options.mtl_basedir = argv[i++];
    options.image_size[0] = std::atoi(argv[i++]);
    options.image_size[1] = std::atoi(argv[i++]);
    options.focal_length = std::atof(argv[i++]);
    options.z_near = std::atof(argv[i++]);
    options.z_far = std::atof(argv[i++]);
    options.principal_point[0] = std::atof(argv[i++]);
    options.principal_point[1] = std::atof(argv[i++]);

    options.extrinsics = Eigen::Matrix4f::Identity();
    for (size_t row = 0; row < 3; row++)
    {
        for (size_t col =0; col < 4; col++)
        {
            options.extrinsics(row, col) = std::atof(argv[i++]);
        }
    }
}

void print_options(const ProgramOptions &options)
{
    std::cout << "Obj file: " << options.obj_file << std::endl;
    std::cout << "mtl_basedir: " << options.mtl_basedir << std::endl;
    std::cout << "image size: "
              << options.image_size[0] << ", " << options.image_size[1]
              << std::endl;
    std::cout << "focal length: " << options.focal_length << std::endl;
    std::cout << "z_near: " << options.z_near << std::endl;
    std::cout << "z_far: " << options.z_far << std::endl;
    std::cout << "principal point: "
              << options.principal_point[0] << ", "
              << options.principal_point[1]
              << std::endl;
    std::cout << "projection matrix: " << std::endl;
    std::cout << "extrinsics: " << std::endl
              << options.extrinsics << std::endl;
}

void initializeOpenGL(int argc, char **argv)
{
    /****************************************/
    /*   Initialize GLUT and create window  */
    /****************************************/

    glutInit(&argc, argv);
    glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowPosition( 50, 50 );
    glutInitWindowSize( 800, 600 );
    auto main_window = glutCreateWindow( "Ignore me" );
    // glutDisplayFunc( myGlutDisplay );



	/****************************************/
    /*          Enable z-buferring          */
    /****************************************/

    glEnable(GL_DEPTH_TEST);
}

void saveDepthBuffer(const std::vector<float> depth_buffer,
                       int width, int height,
                       const std::string filename)
{
    assert(width * height == depth_buffer.size());
    std::vector<unsigned char> char_buffer(depth_buffer.size());
    bool valid = false;
    for (size_t i = 0; i < depth_buffer.size(); i++)
    {
        if (depth_buffer[i])
        {
            valid = true;
        }
        char_buffer[i] = depth_buffer[i] * 255;
    }
    std::cout << "depth valid: " << valid << std::endl;
    cv::Mat mat(cv::Size(width, height), CV_8UC1, char_buffer.data());
    // OpenGL is column major, OpenCV is row major
    cv::flip(mat, mat, 0);
    cv::imwrite(filename, mat);
}

int main(int argc, char **argv)
{
    using namespace p2mpp_depth_renderer;
    using ProjectionParameters =
        p2mpp_depth_renderer::offscreen_rendering::ProjectionParameters;
    ProgramOptions options;
    get_options(argc, argv, options);
    print_options(options);
    ProjectionParameters projection_parameters(
        options.extrinsics, options.focal_length, options.image_size,
        options.z_near, options.z_far, MESH_SCALE_FACTOR
    );

    Mesh mesh;
    auto mesh_status = mesh.loadObj(options.obj_file, options.mtl_basedir);
    if (!mesh_status)
    {
        std::cerr << "Could not read mesh" << std::endl;
        exit(1);
    }
    mesh.printMeshStats();

	initializeOpenGL(argc, argv);
    std::cout << "OpenGL successfully initialize" << std::endl;
    GlContext gl_context;
    gl_context.initialize();
    const auto float_depth_buffer = gl_context.render(
        mesh.getVertices(), mesh.getFaces(), projection_parameters
    );
    saveDepthBuffer(
        float_depth_buffer, options.image_size[0], options.image_size[1],
        "/tmp/depth_buffer.png"
    );

    return 0;
}
