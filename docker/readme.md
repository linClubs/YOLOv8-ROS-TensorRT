~~~python
# 构建镜像
docker build -f Dockerfile -t yolo:v1 .

# 新建容器
sudo docker run -it --gpus all -v ~/share:/root/share --name test yolo:v1 /bin/bash
~~~