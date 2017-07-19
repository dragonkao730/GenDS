


FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS = " ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV_LIBS = " ${OpenCV_LIBS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES("/usr/local/include/eigen3")

ADD_EXECUTABLE(GenDS
    align_parallax.cpp
    alignment.cpp
    mesh.cpp
    optimisor_o.cpp)

TARGET_LINK_LIBRARIES(GenDS ${OpenCV_LIBS})





