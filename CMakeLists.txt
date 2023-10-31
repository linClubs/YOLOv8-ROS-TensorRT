cmake_minimum_required(VERSION 3.5)
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

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        include
)

link_directories(
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES} 
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

# ------------ros-----------------------

add_executable(detect src_ros/detect.cpp)
target_link_libraries(detect 
        nvinfer
        nvinfer_plugin
        cudart
        ${OpenCV_LIBRARIES} 
)

ament_target_dependencies(detect
  rclcpp 
  std_msgs
  sensor_msgs
  cv_bridge
)

add_executable(pose src_ros/pose.cpp)
target_link_libraries(pose
        nvinfer 
        nvinfer_plugin 
        cudart 
        ${OpenCV_LIBS}
)

ament_target_dependencies(pose
  rclcpp 
  std_msgs
  sensor_msgs
  cv_bridge
)

add_executable(segment src_ros/segment.cpp)
target_link_libraries(segment 
        nvinfer 
        nvinfer_plugin 
        cudart 
        ${OpenCV_LIBS}
)


ament_target_dependencies(segment
  rclcpp 
  std_msgs
  sensor_msgs
  cv_bridge
)

# c程序需要加下面2句 
install(TARGETS
  detect
  pose
  segment
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION 
  share/${PROJECT_NAME})
