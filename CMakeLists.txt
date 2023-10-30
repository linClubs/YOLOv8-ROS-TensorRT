cmake_minimum_required(VERSION 3.1)
project(yolov8_trt)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64)

# TensorRT  # TensorRT_ROOT的路径设置成自己的
set(TensorRT_ROOT /home/lin/software/TensorRT-8.5.3.1)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARIES ${TensorRT_ROOT}/targets/x86_64-linux-gnu/lib)

# message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
# message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

find_package(CUDA REQUIRED)
# find_package(TensorRT REQUIRED)
# OpenCV
find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  cv_bridge
  sensor_msgs
)

catkin_package(
)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        include

        ${catkin_INCLUDE_DIRS}
)

link_directories(
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES} 
)

# -----------without ros---------------
# add_executable(detect src/detect.cpp)
# # target_link_directories(detect PUBLIC
# #         ${CUDA_LIBRARIES}
# #         ${TensorRT_LIBRARIES}
# #         ${OpenCV_LIBRARIES} 
# # )
# target_link_libraries(detect 
#         nvinfer
#         nvinfer_plugin
#         cudart
#         ${OpenCV_LIBRARIES} 
        
# )

# add_executable(pose src/pose.cpp)
# target_link_libraries(pose PRIVATE 
#         nvinfer 
#         nvinfer_plugin 
#         cudart 
#         ${OpenCV_LIBS}
# )

# add_executable(segment src/segment.cpp)
# target_link_libraries(segment PRIVATE 
#         nvinfer 
#         nvinfer_plugin 
#         cudart 
#         ${OpenCV_LIBS}
# )

# ------------ros-----------------------

add_executable(detect src_ros/detect.cpp)
target_link_libraries(detect 
        nvinfer
        nvinfer_plugin
        cudart
        ${OpenCV_LIBRARIES} 
        ${catkin_LIBRARIES}     
)

add_executable(pose src_ros/pose.cpp)
target_link_libraries(pose PRIVATE 
        nvinfer 
        nvinfer_plugin 
        cudart 
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}   
)

add_executable(segment src_ros/segment.cpp)
target_link_libraries(segment PRIVATE 
        nvinfer 
        nvinfer_plugin 
        cudart 
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}    
)