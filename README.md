
+ ros1-noetic版本、CUDA-11.3、TensorRT-8.5

下载官方权重详情见[readme](https://github.com/ultralytics/ultralytics/blob/main/README.zh-CN.md)

+ 没有python环境，可以直接用`onnx`格式转`engine`格式，无需`pt`文件
+ `onnx`可以跨平台使用，`engine`跟`tensorrt`有关，不能跨平台通用

# 1 onnx2engine
~~~python
# 1 进入tensorrt空间
cd tensorRT/bin

# 2 运行trtexec脚本 onnx生成engine
/tensorRT/bin/trtexec --onnx=xxx.onnx --saveEngine=xxx.engine 

# pc上
~/software/TensorRT-8.5.3.1/bin/trtexec --onnx=yolov8l-seg.onnx --saveEngine=yolov8l-seg.engine

# orin上
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n-seg.onnx --saveEngine=yolov8n-seg.engine
~~~

# 2 编译运行

1. 编译
~~~
# 1 拉取源码
mkdir -p yolo_ws/src && cd yolo_ws/src
git clone 

# 2 安装ros依赖
cd yolo_ws
rosdep install -r -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
~~~

2. 修改tensorRT, cuda环境配置

~~~c
# ----------x86-----------
# cuda
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64)

# TensorRT  # TensorRT_ROOT的路径设置成自己的
set(TensorRT_ROOT /home/lin/software/TensorRT-8.5.3.1)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARIES ${TensorRT_ROOT}/targets/x86_64-linux-gnu/lib)

#-----------arm-orin-------
set(CUDA_INCLUDE_DIRS /usr/local/cuda-11.4/include)
set(CUDA_LIBRARIES /usr/local/cuda-11.4/targets/aarch64-linux/lib)

set(TensorRT_INCLUDE_DIRS /usr/include)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu/)

message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")

message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")
message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
~~~

3. 编译

~~~python
cd yolo_ws
catkin_make
~~~

4. 修改对应的launch文件的参数即可

~~~xml
topic_img 待检测的图像话题
topic_res_img 结果图像发布话题
weight_name 权重名称，权重放入weights目录下
~~~

5. 运行

~~~python
# detect
roslaunch yolov8_trt segment.launch

# pose
roslaunch yolov8_trt segment.launch

# segment
roslaunch yolov8_trt segment.launch
~~~
