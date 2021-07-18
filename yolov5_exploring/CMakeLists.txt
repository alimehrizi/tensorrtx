cmake_minimum_required(VERSION 3.2)

project(trt-detector)


set(CMAKE_CXX_STANDARD 11)



find_package(OpenCV PATHS /home/altex/OPENCV-4.5.0 NO_DEFAULT_PATH)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/yolov5)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different

# cuda
include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-7.2.2.3/include)
link_directories(/usr/local/TensorRT-7.2.2.3/lib)



set(YOLO_SRC ${PROJECT_SOURCE_DIR}/yolov5/yololayer.cu
               ${PROJECT_SOURCE_DIR}/yolov5/yololayer.h
               ${PROJECT_SOURCE_DIR}/yolov5/common.hpp
               ${PROJECT_SOURCE_DIR}/yolov5/cuda_utils.h
               ${PROJECT_SOURCE_DIR}/yolov5/gen_wts.py
               ${PROJECT_SOURCE_DIR}/yolov5/logging.h
               ${PROJECT_SOURCE_DIR}/yolov5/utils.h
               ${PROJECT_SOURCE_DIR}/yolov5/yolov5.cpp
               ${PROJECT_SOURCE_DIR}/yolov5/yolov5.h)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(deep_cores SHARED  ${YOLO_SRC})
target_link_libraries(deep_cores ${OpenCV_LIBS})
target_link_libraries(deep_cores cudart)
target_link_libraries(deep_cores /usr/local/TensorRT-7.2.2.3/lib/libnvinfer.so)
target_link_libraries(deep_cores ${CUDA_LIBRARIES} )





#set(CMAKE_CXX_FLAGS  "-DDEBUG")
#set(BOOST_ROOT /usr/include/boost)
find_package(Boost COMPONENTS system filesystem log log_setup REQUIRED)
set(PROJECT_LIB ${PROJECT_LIB} ${Boost_LOG_LIBRARY} ${Boost_LOG_SETUP_LIBRARY} )
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -fopenmp  -DBOOST_LOG_DYN_LINK")


include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SOURCE_FILES src/*.cpp src/*.cu)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} deep_cores)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} -Wl,--start-group  ${PROJECT_LIB} -Wl,--end-group)

#add_definitions(-O2 -pthread)

