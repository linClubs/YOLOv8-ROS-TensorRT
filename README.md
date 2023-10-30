
下载官方权重详情见[readme](https://github.com/ultralytics/ultralytics/blob/main/README.zh-CN.md)

+ 没有python环境，可以直接用`onnx`格式转`engine`格式，无需`pt`文件
+ `onnx`可以跨平台使用，`engine`跟`tensorrt`有关，不能跨平台通用

# 1 onnx2engine
~~~python
# 1 进入tensorrt空间
cd tensorRT/bin

# 2 运行trtexec脚本 onnx生成engine
/tensorRT/bin/trtexec --onnx=xxx.onnx --saveEngine=xxx.engine 
~~~

# 2 编译运行

1. 编译
~~~
mkdir build
cd build
cmake ..
make -j
~~~


2. 运行
~~~
./detect
./pose
./segment
~~~