cmake_minimum_required(VERSION 2.6)

project(yolov5)


set(CMAKE_CXX_STANDARD 11)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-7.2.2.3/include)
link_directories(/usr/local/TensorRT-7.2.2.3/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins /usr/local/TensorRT-7.2.2.3/lib/libnvinfer.so)
target_link_libraries(myplugins cudart)
target_link_libraries(myplugins ${CUDA_LIBRARIES} )

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 /usr/local/TensorRT-7.2.2.3/lib/libnvinfer.so)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

