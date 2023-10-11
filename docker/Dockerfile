
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 1 setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# 2 apt换源
RUN echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list \ 
    && echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

# 3 更新源和安装wget和git
RUN apt update && apt install wget cmake -y

# ImportError: libGL.so.1:找不到
RUN apt-get install ffmpeg libsm6 libxext6  -y

# 4 安装conda并初始化
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b     \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh   \
    && ~/miniconda3/bin/conda init

# 5 conda和pip换国内源
RUN ~/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  \
    && ~/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  \
    && ~/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/  \
    && ~/miniconda3/bin/conda config --set show_channel_urls yes  \
    && mkdir ~/.pip && touch ~/.pip/pip.conf \
    && echo "[global]" > ~/.pip/pip.conf  \
    && echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple/" >> ~/.pip/pip.conf

# 6 其他依赖
RUN apt install git vim sudo kmod
RUN apt install libopencv-dev

WORKDIR /root