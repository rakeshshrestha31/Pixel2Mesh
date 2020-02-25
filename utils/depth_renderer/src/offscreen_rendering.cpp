/**
 * @file offscreen_rendering.cpp
 * @author Rakesh Shrestha, rakeshs@sfu.ca
 */
#include <p2mpp_depth_renderer/offscreen_rendering.h>

#include <GL/gl.h>

namespace p2mpp_depth_renderer
{

void offscreen_rendering::mGLRender(
        std::vector<Eigen::Vector3f> vertices,
        std::vector< Eigen::Matrix<GLushort,-1,3> > faces,
        std::vector<Eigen::Vector3f> cams,
        float scale, bool inverse
)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // if (!inverse)
    //     gluLookAt(cam[0], cam[1], cam[2], 0, 0, 0, 0, 1e-10, -1);
    // else
    //     gluLookAt(cam[0], cam[1], cam[2], 0, 0, 0, 0, -1, 0);

    // glTranslatef(0, 0, -2.2);
    // glRotatef(90, cam[0], cam[1], cam[2]);


    for (auto cam: cams)
    {
        glRotatef(90, cam[0], cam[1], cam[2]);
    }

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glBegin(GL_TRIANGLES);

    int num = vertices.size();
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < faces[i].rows(); j++)
        {
            auto p0 = vertices[i].row(faces[i](j, 0));
            auto p1 = vertices[i].row(faces[i](j, 1));
            auto p2 = vertices[i].row(faces[i](j, 2));

            p0 *= scale;
            p1 *= scale;
            p2 *= scale;

            glVertex3fv(p0.data());
            glVertex3fv(p1.data());
            glVertex3fv(p2.data());
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
        std::vector<Eigen::Vector3f> vertices,
        std::vector<Eigen::Matrix<GLushort, -1, 3>> faces,
        std::vector<Eigen::Vector3f> cams,
        std::array<GLsizei, 2> image_size,
        float scale, bool inverse
)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 3);
    glViewport(0, 0, image_size[0], image_size[1]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // gluLookAt(1, 0, 0, 0, 0, 0, 0, 1, 0);
    mGLRender(vertices, faces, cams, scale, inverse);

    std::vector<float> depth_z(image_size[0] * image_size[1]);

    glReadBuffer(GL_BACK);
    glReadPixels(
        0, 0, image_size[0], image_size[1], GL_DEPTH_COMPONENT, GL_FLOAT,
        (GLvoid*)depth_z.data()
    );
    glPopAttrib();

    return depth_z;
}
} // endnamespace p2mpp_depth_renderer

