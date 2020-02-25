/**
 * @file offscreen_rendering.cpp
 * @author Rakesh Shrestha, rakeshs@sfu.ca
 */
#include <p2mpp_depth_renderer/offscreen_rendering.h>

namespace p2mpp_depth_renderer
{

void offscreen_rendering::mGLRender(
        const Eigen::Matrix<float, -1, 3> &vertices,
        const Eigen::Matrix<GLushort, -1, 3> &faces,
        const ProjectionParameters &projection_parameters
)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(projection_parameters.T_cam_world.data());

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glBegin(GL_TRIANGLES);

    Eigen::Matrix<float, -1, 3> scale_vertices = vertices;
    scale_vertices *= projection_parameters.scale;
    int num_faces = faces.rows();
    for (int f_idx = 0; f_idx < num_faces; f_idx++)
    {
        for (int fv_idx = 0; fv_idx < 3; fv_idx++)
        {
            const auto v_idx = faces(f_idx, fv_idx);
            const auto p = vertices.row(v_idx);
            glVertex3fv(p.data());
        }
    }
    glEnd();
    glFlush(); // remember to flush GL output!
}

/**
 *
 * @param vertices vector of vertices for each segment (the vertices are stored a float vectors for each segment)
 * @param faces vector of indices that make faces from the vertices
 * @param cams
 * @param inverse
 * @return
 */
std::vector<float> offscreen_rendering::render(
        const Eigen::Matrix<float, -1, 3> &vertices,
        const Eigen::Matrix<GLushort, -1, 3> &faces,
        const ProjectionParameters &projection_parameters
)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(
        projection_parameters.fovy,
        projection_parameters.image_size[0]
            / projection_parameters.image_size[1],
        projection_parameters.z_near, projection_parameters.z_far
    );
    glViewport(
        0, 0, projection_parameters.image_size[0],
        projection_parameters.image_size[1])
    ;

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // gluLookAt(1, 0, 0, 0, 0, 0, 0, 1, 0);
    mGLRender(vertices, faces, projection_parameters);

    std::vector<float> depth_z(
        projection_parameters.image_size[0] \
            * projection_parameters.image_size[1]
    );

    glReadBuffer(GL_BACK);
    glReadPixels(
        0, 0, projection_parameters.image_size[0],
        projection_parameters.image_size[1],
        GL_DEPTH_COMPONENT, GL_FLOAT, (GLvoid*)depth_z.data()
    );
    glPopAttrib();

    return depth_z;
}
} // endnamespace p2mpp_depth_renderer

