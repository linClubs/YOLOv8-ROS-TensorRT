
# `YOLOv8-TensorRT-CPP`

# 1. 环境配置

## 1.1 显卡驱动 `cuda` `cudnn`安装

[安装参考链接](https://blog.csdn.net/h904798869/article/details/131719404)

## 1.2 基础环境安装
~~~python
# 1 创建python虚拟环境
conda create -n yolov8 python=3.8
# 2 激活虚拟环境

# 3 安装torch-1.10.0 torchvision==0.11
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# 4 其他依赖
pip install numpy==1.23.5 onnx==1.14.1  onnxsim opencv-python==4.5.4.58 ultralytics onnxruntime==1.16.0
# mpmath, flatbuffers, sympy, humanfriendly, coloredlogs

# 5 tensorrt安装参考下面
~~~

+ `TensorRT`安装

[安装参考安装TensorRT部分](https://blog.csdn.net/h904798869/article/details/131719404)

[下载地址](https://link.csdn.net/?target=https%3A%2F%2Fdeveloper.nvidia.com%2Fnvidia-tensorrt-download) 需要注册账号登录才能下载

~~~python
# 1 解压TensorRT得到TensorRT-8.5.3.1目录
tar -xf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz

# 2 安装python版tensorrt
# 2.1 进入TensorRT-8.5.3.1/python目录
cd TensorRT-8.5.3.1/python
# 2.2 安装python3.8支持的tensorrt, 选择cp38即可
pip install tensorrt-8.5.3.1-cp38-none-linux_x86_64.whl 

# 3 c++ 版TensorRT直接使用这个文件, 
# 修改CMakeLists.txt的TensorRT_ROOT路径, 改成自己对应TensorRT所在的路径即可使用
set(TensorRT_ROOT /path/to/TensorRT-8.5.3.1)
~~~

# 1.3 报错汇总

1. 运行`python`脚本报错找不到`TensorRT`相关的库文件

+ 修改方法2：
将缺的`TensorRT`相关的库文件放进`/usr/lib`

~~~python
# 1 错误1
ImportError: libnvinfer.so.8: cannot open shared object file: No such file or directory
# 修改将libnvinfer.so.8放入/usr/lib
sudo cp TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib/libnvinfer.so.8 /usr/lib

# 2 错误2
mportError: libnvonnxparser.so.8: cannot open shared object file: No such file or directory
# 修改将libnvonnxparser.so.8放入/usr/lib
sudo cp TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib/libnvonnxparser.so.8 /usr/lib

# 错误3
libnvparsers.so.8：cannot open shared object file: No such file or directory
# 修改将libnvparsers.so.8放入/usr/lib
sudo cp TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib/libnvparsers.so.8 /usr/lib

# 错误4 
libcudnn.so.8 cannot open shared object file
# 把/usr/local/cuda/lib64/libcudnn.so.8文件放入/usr/lib

# 错误5 c++推理代码运行报错
fatal error: NvInferPlugin.h: No such file or directory
# 修改CMakeLists.txt的TensorRT_ROOT路径
set(TensorRT_ROOT /root/share/TensorRT-8.5.3.1)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARIES ${TensorRT_ROOT}/lib)

# 报错6
TypeError: pybind11::init(): factory function returned nullptr
~~~


# 2 pt2onnx

`pt`文件转`onnx`文件

## 2.1 下载`pt`文件

下载`pt`文件详情见[官方地址readme](https://github.com/ultralytics/ultralytics/blob/main/README.md)

可以直接点击下面链接下载`yolov8n`相关模型：

[yolov8n-det](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

[yolov8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)

[yolov8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)


## 1.2 `det and seg` 模型转换
~~~python
# 1 进入scripts目录
cd scripts

# 2 det
python export-det.py --weights ../weights/yolov8n.pt --iou-thres 0.65 --conf-thres 0.2 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0

# 3 seg
python export-seg.py --weights ../weights/yolov8n-seg.pt --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0

# --topk最大检测框数 
# -input-shape输入尺寸
~~~

## 1.3 `pose`模型转换

1. `scripts`目录下新建一个转换脚本`export-pose.py`文件，填入下面内容

~~~python
from ultralytics import YOLO
# Load a model
model = YOLO("../weights/yolov8n-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success
~~~

2. 运行`export-pose.py`生成`onnx`文件

~~~python
python export-pose.py
~~~

生成的`onnx`与对应的`pt`在同级目录


# 2 `onnx2engine`

+ `onnx`文件转`engine`格式文件

~~~python
# 1 进入scripts目录
cd scripts

# 2 det
python build.py --weights ../weights/yolov8n.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16  --device cuda:0

# 3 seg, 增加--seg参数
python build.py --weights ../weights/yolov8n-seg.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16  --device cuda:0 --seg

# 4 pose 使用tensorRT
cd tensorRT/bin
./trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose.engine 
~~~

+ 等待时间比较久，只要电脑没开，都正常

# 3 `c++`模型推理

1. 配置`TensorRT`环境变量

+ c++编译前,需要修改`CMakeLists.txt`中`TensorRT_ROOT`路径, 改成自己对应TensorRT所在的路径即可使用

~~~c
set(TensorRT_ROOT /path/to/TensorRT-8.5.3.1)
~~~

2. 编译源码

+ `src/detect/end2end`检测代码为例

~~~python
# 1 进入目录
cd src/detect/end2end

# 2 cmake编译
mkdir build && cd build && cmake .. && make -j

# 3 pose代码就进入src/pose/normal进行cmake编译
~~~

3. `inference`推理

+ 注意权重路径和待检测图像路径给正确即可

~~~python
# det-python版本  需要进入scripts路径运行
python3 infer-det.py --engine yolov8s.engine --imgs data --show --out-dir outputs
--device cuda:0

# 1 检测 det-c++版本
# 1.1 infer image
./yolov8 yolov8n.engine data/bus.jpg
# 1.2 infer images
./yolov8 yolov8n.engine data
# 1.3 infer video
./yolov8 yolov8n.engine data/test.mp4 # the video path

# 2 分割 seg
./yolov8-seg weights/yolov8n-seg.engine data/zidane.jpg 

# 3 姿态预测 pose
./yolov8-pose weights/yolov8n-pose.engine data/bus.jpg
~~~

---