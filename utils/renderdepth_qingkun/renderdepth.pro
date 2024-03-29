TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -std=c++0x -fopenmp

SOURCES += \
        main.cpp

INCLUDEPATH += $$PWD/trimesh2/include

INCLUDEPATH += /usr/include \
               /usr/include/opencv \
               /usr/include/opencv2 \
               /usr/include/eigen3

LIBS += $$PWD/trimesh2/lib.Linux64/libtrimesh.a
LIBS += -lglut -lGLU -lGL -lGLEW
LIBS += /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
        /usr/lib/x86_64-linux-gnu/libopencv_core.so \
        /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so \
        /usr/lib/x86_64-linux-gnu/libopencv_photo.so \

LIBS += -fopenmp
