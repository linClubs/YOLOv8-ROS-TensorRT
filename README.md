
https://ghproxy.com/111
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hyl

# YOLOv8-TensorRT

# 1. 环境配置

# 1. 基础环境安装
~~~python
# 1 创建python虚拟环境
conda create -n yolov8 python=3.8
# 2 激活虚拟环境

# 3 安装torch-1.10.0 torchvision==0.11
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html


pip install numpy==1.23.5 onnx==1.14.1 opencv-python == 4.5.4.58 ultralytics onnxruntime==1.16.0
# mpmath, flatbuffers, sympy, humanfriendly, coloredlogs

# 1.2 报错

1. 运行`c++`推理报错找不到`TensorRT`相关的库文件

+ 修改方法1：
在`CMakeLists.txt`加入
~~~python
set(TensorRT_ROOT /path/to/TensorRT)   #  /path/to/TensorRT换成自己的tensorRT路径
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBS ${TensorRT_ROOT}/lib/)

include_directories(
   ${TensorRT_INCLUDE_DIRS}
)

link_directories(
   ${TensorRT_LIBS}
)
~~~


+ 修改方法2：
将缺的TensorRT相关的库文件放进`/usr/lib`

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
~~~


# 2 pt2onnx

pt文件转onnx文件

## 2.1 下载`pt`文件

下载`pt`文件详情见[官方地址readme](https://github.com/ultralytics/ultralytics/blob/main/README.md)

可以直接点击下面链接下载`yolov8n`相关模型：

[yolov8n-det](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

[yolov8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)

[yolov8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)


## 1.2 det and seg 模型转换
~~~python
# 1 det
python export-det.py --weights weights/yolov8n.pt --iou-thres 0.65 --conf-thres 0.2 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0

# 2 seg
python export-seg.py --weights weights/yolov8n-seg.pt --opset 11 --sim --inpu
t-shape 1 3 640 640 --device cuda:0

# --topk最大检测框数 
# -input-shape输入尺寸
~~~

## 1.3 pose模型转换

1. 新建一个转换脚本`export-pose.py`，填入下面内容

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


# 2 onnx2engine

+ onnx文件转engine格式文件

~~~python
# 1 det
python3 build.py --weights weights/yolov8n.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16  --device cuda:0

# 2 seg, 增加--seg参数
python build.py --weights weights/yolov8n-seg.onnx --iou-thres 0.65 --conf-thres 0.25 --topk 100 --fp16  --device cuda:0 --seg

# 3 pose 使用tensorRT
cd tensorRT/bin
./trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose.engine 
~~~

+ 等待时间比较久，只要电脑没开，都正常

# 3 c++模型推理

1. det-inference检测推理

~~~python
# 1 det-c++版本
mv weights/yolov8n.engine csrc/detect/end2end/build/
cd csrc/detect/end2end/build

# det-python版本
python3 infer-det.py --engine yolov8s.engine --imgs data --show --out-dir outputs
--device cuda:0

# 1.1 infer image
./yolov8 yolov8n.engine data/bus.jpg
# 1.2 infer images
./yolov8 yolov8n.engine data
# 1.3 infer video
./yolov8 yolov8n.engine data/test.mp4 # the video path

# 2 seg
./yolov8-seg weights/yolov8n-seg.engine data/zidane.jpg 

# 3 pose
./yolov8-pose ../../weights/yolov8n-pose.engine ../../data/bus.jpg
~~~

---