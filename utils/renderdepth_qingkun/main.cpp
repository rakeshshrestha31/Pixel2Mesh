// #include <windows.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <TriMesh.h>
#include <Eigen/Eigen>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <string.h>
#include <strings.h>

using namespace std;
using namespace trimesh;
using namespace cv;

double focal;
int width, height;

TriMesh *mesh;
vector<double> intrinsic;
vector<Eigen::Matrix<double, 3, 4>> extrinsic_list;

vector<std::string> out_names;
double min_distance, max_distance;
double depth_unit;
int currentView = 0;


//vec3 centerPos(0.0, 0.0, 0.0);
//string imageName;
//string imagePath;
//string txtPath;
//vector<vec3> camPosList;

void display()
{
	if (currentView >= extrinsic_list.size()) exit(0);
	int w = glutGet(GLUT_WINDOW_WIDTH);
	int h = glutGet(GLUT_WINDOW_HEIGHT);

    // glClearDepth(0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	double ar = w / static_cast< double >(h);
    double fov = 2.0 * std::atan(h / 2.0 / focal) * 180 / M_PI;
    std::cout << "width,height: " << w << " x " << h << std::endl;
    std::cout << "fiew of view: " << fov << std::endl;
    std::cout << "aspect ratio:  " << ar << std::endl;
    std::cout << "min distance: " << min_distance << std::endl;
    std::cout << "max distance: " << max_distance << std::endl;

    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(fov, ar, min_distance, max_distance); // simulate kinect

    Eigen::Matrix3d m_inverse = extrinsic_list[currentView].block<3,3>(0, 0);
    m_inverse = m_inverse.transpose().eval();
    Eigen::Vector3d pose = -m_inverse * extrinsic_list[currentView].col(3);

	Eigen::Vector3d eye(pose(0), pose(1), pose(2));
    Eigen::Vector3d center = eye + m_inverse.col(2);
    Eigen::Vector3d up = -m_inverse.col(1);

    std::cout << "eye:    " << eye.transpose() << std::endl;
    std::cout << "center: " << center.transpose() << std::endl;
    std::cout << "up:     " << up.transpose() << std::endl;

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2]);

	// render mesh
    glColor3ub(255, 0, 0);
	for (int it = 0; it < mesh->faces.size(); it++)
	{
		glBegin(GL_TRIANGLES);
		for (int iv = 0; iv < 3; iv++)
		{
			glNormal3f(mesh->normals[mesh->faces[it][iv]][0], mesh->normals[mesh->faces[it][iv]][1], mesh->normals[mesh->faces[it][iv]][2]);
			glVertex3f(mesh->vertices[mesh->faces[it][iv]][0], mesh->vertices[mesh->faces[it][iv]][1], mesh->vertices[mesh->faces[it][iv]][2]);
		}
		glEnd();
	}
	glPopMatrix();
	vector< GLfloat > depth(w * h, 0);
	glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]); // read depth buffer
    cv::Mat depth_image(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), CV_16UC1);
	for (int i = 0; i < depth_image.rows; i++)
	{
		for (int j = 0; j < depth_image.cols; j++)
		{
            float depth_val = depth[i*depth_image.cols + j];
			depth_val = (2.0 * min_distance * max_distance) / (max_distance + min_distance - (2.0f * depth_val - 1) * (max_distance - min_distance));
//			depth[i*depth_image.cols + j] = (depth[i*depth_image.cols + j] - min_distance) / (max_distance - min_distance);
            if (depth_val > min_distance && depth_val < max_distance)
                depth_image.at<u_int16_t>(i, j) = std::round(depth_val * depth_unit);
            else
                depth_image.at<u_int16_t>(i, j)  = 0;
		}
	}
	cv::Mat flipped(depth_image);
	cv::flip(depth_image, flipped, 0);

	string currentName = out_names[currentView] + ".png";
    std::cout << currentName << std::endl;
	cv::imwrite(currentName, depth_image);
	currentView++;
	glutSwapBuffers();
}

void timer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(16, timer, 0);
}

int main(int argc,char **argv)
{
	std::string mesh_path = argv[1];
    // each line contains width, height, focal, min_distance, max_distance, 12 extrinsic number, and output path
    std::string params_list_file = argv[2];
    std::cout << "mesh: " << mesh_path << std::endl;
    std::cout << "params: " << params_list_file << std::endl;

    mesh = TriMesh::read(mesh_path);
	ifstream fin(params_list_file);
    std::string line;
    while(std::getline(fin, line)) {
        if(line.length() == 0) continue;

        istringstream in(line);
        in >> width >> height >> focal >> min_distance >> max_distance >> depth_unit;
        Eigen::Matrix<double, 3, 4> m;
        for(int i = 0; i < 12; i++) {
            in >> m(i / 4, i % 4);
        }

        std::string out_name;
        in >> out_name;
        out_names.push_back(out_name);
        extrinsic_list.push_back(m);
    }

	mesh->need_normals();
	mesh->need_bsphere();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    printf("hahaha\n");
	glutInitWindowSize(width, height);


	glutCreateWindow("GLUT");

	glewInit();

	glutDisplayFunc(display);
	glutTimerFunc(0, timer, 0);
	glEnable(GL_DEPTH_TEST);
	glutMainLoop();
	return 0;
}
