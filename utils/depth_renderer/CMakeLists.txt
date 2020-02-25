cmake_minimum_required(VERSION 3.0.2)
project(p2mpp_depth_renderer)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
add_compile_options(-std=c++14)

# opengl stuffs for multi-context
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
add_definitions(${GLEW_DEFINITIONS})
if(NOT GLEW_FOUND)
    message(Error " GLEW not found")
endif(NOT GLEW_FOUND)

# glfw3
find_package(PkgConfig)
pkg_check_modules(GLFW REQUIRED glfw3)

include_directories(${GLFW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
add_definitions(${GLFW_DEFINITIONS})

if(NOT GLFW_FOUND)
    message(Error " GLFW not found")
endif(NOT GLFW_FOUND)

# glm
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
link_directories(${GLM_LIBRARY_DIRS})
add_definitions(${GLM_DEFINITIONS})

if(NOT GLM_FOUND)
    message(Error " GLM not found")
endif(NOT GLM_FOUND)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tinyobjloader)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME} SHARED
    src/gl_context.cpp
    src/offscreen_rendering.cpp
    src/mesh.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
	${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLM_LIBRARIES}
    ${EIGEN3_LIBS}
    tinyobjloader
)

add_executable(
    render_depth src/main.cpp
)
target_link_libraries(
    render_depth
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)
