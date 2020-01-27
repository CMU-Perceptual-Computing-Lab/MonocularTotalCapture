FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="Necati Cihan Camgoz <n.camgoz@surrey.ac.uk>"

ENV PATH=/usr/local/cuda-9.0/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}
RUN ln -s /usr/local/cuda-9.0/lib64/libcudart.so /usr/lib/libcudart.so

ENV CUDA_ARCH_BIN "50 52 60 61"
ENV CUDA_ARCH_PTX "50 52 60 61"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ffmpeg \
    freeglut3 \
    freeglut3-dev \
    glew-utils \
    libglew-dev \
    libatlas-base-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libprotobuf-dev \
    libsnappy-dev \
    libx11-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxi-dev \
    libxmu-dev \
    libblas-dev \ 
    libxinerama-dev \
    libxcursor-dev \
    libglm-dev \
    llvm-6.0 \
    mesa-common-dev \
    mesa-utils \
    nano \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    python-scipy \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    python3-scipy \
    software-properties-common \
    unzip \
    xorg-dev && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get autoremove -y && apt-get autoclean -y

# PYTHON
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools \
    wheel \
    tensorflow-gpu==1.12.0 \
    opencv-python \
    scikit-image \
    Mako \
    matplotlib \
    numpy \
    protobuf

# LIBIGL
ENV LIBIGL_ROOT=/opt/libigl
WORKDIR ${LIBIGL_ROOT}
RUN git clone https://github.com/libigl/libigl.git . && \
    mkdir build && cd build && \
    cmake .. && \
    make all -j8

# CERES
ENV CERES_ROOT=/opt/ceres
WORKDIR ${CERES_ROOT}
RUN git clone -b 1.13.0 https://ceres-solver.googlesource.com/ceres-solver.git . && \
    mkdir build && cd build && \
    cmake .. && \
    make all -j8 && \
    make test && \
    make install

# OPENCV
ENV OPENCV_ROOT=/opt/opencv
WORKDIR ${OPENCV_ROOT}
RUN git clone -b 2.4.13.6 https://github.com/opencv/opencv.git .
RUN wget https://raw.githubusercontent.com/yjxiong/anet2016-cuhk/docker_server/opencv_cuda9.patch
RUN git apply opencv_cuda9.patch
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D BUILD_SHARED_LIBS=ON \
    -D WITH_CUBLAS=1 .. &&\
    make all -j8 && \
    make install
ENV PATH=/opt/opencv/build/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/opencv/build/lib:${LD_LIBRARY_PATH}

# CAFFE
RUN apt-get install 
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
RUN git clone -b 1.0 https://github.com/CMU-Perceptual-Computing-Lab/caffe.git .
RUN cd python && \
    pip3 install -r requirements.txt && \
    pip install -r requirements.txt && \
    cd ..
RUN cp Makefile.config.example Makefile.config
RUN sed -i 's/# USE_CUDNN := 1/USE_CUDNN := 1/' Makefile.config
RUN sed -i 's/CUDA_DIR := \/usr\/local\/cuda/CUDA_DIR := \/usr\/local\/cuda-9.0/' Makefile.config
RUN sed -i 's/# PYTHON_LIBRARIES := boost_python3 python3.5m/PYTHON_LIBRARIES := boost_python3 python3.5m/' Makefile.config
RUN sed -i '/CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \\/d' Makefile.config
RUN sed -i '/		-gencode arch=compute_20,code=sm_21 \\/d' Makefile.config
RUN sed -i '/		-gencode arch=compute_30,code=sm_30 \\/d' Makefile.config
RUN sed -i '/		-gencode arch=compute_35,code=sm_35 \\/d' Makefile.config
RUN sed -i 's/		-gencode arch=compute_50,code=sm_50 \\/CUDA_ARCH := -gencode arch=compute_50,code=sm_50 \\/' Makefile.config
RUN sed -i 's/PYTHON_INCLUDE := \/usr\/include\/python2.7 \\/# PYTHON_INCLUDE := \/usr\/include\/python2.7 \\/' Makefile.config
RUN sed -i 's/		\/usr\/lib\/python2.7\/dist-packages\/numpy\/core\/include/# 		\/usr\/lib\/python2.7\/dist-packages\/numpy\/core\/include/' Makefile.config
RUN sed -i 's/# PYTHON_LIBRARIES := boost_python3 python3.5m/PYTHON_LIBRARIES := boost_python3 python3.5m/' Makefile.config
RUN sed -i 's/# PYTHON_INCLUDE := \/usr\/include\/python3.5m \\/PYTHON_INCLUDE := \/usr\/include\/python3.5m \\/' Makefile.config
RUN sed -i 's/#                 \/usr\/lib\/python3.5\/dist-packages\/numpy\/core\/include/                 \/usr\/lib\/python3.5\/dist-packages\/numpy\/core\/include/' Makefile.config
RUN mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" .. && \
    make all -j8 && \
    make install
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# OPENPOSE
ENV OPENPOSE_ROOT=/opt/openpose
WORKDIR $OPENPOSE_ROOT
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git . && \
    sed -i 's/set(Caffe_known_gpu_archs "${KEPLER} ${MAXWELL} ${PASCAL} ${VOLTA} ${TURING}")/set(Caffe_known_gpu_archs "${KEPLER} ${MAXWELL} ${PASCAL}")/' /opt/openpose/cmake/Cuda.cmake && \
    mkdir build && cd build && \
    cmake -DCaffe_INCLUDE_DIRS=$CAFFE_ROOT/build/install/include \
          -DCaffe_LIBS=$CAFFE_ROOT/build/install/lib/libcaffe.so \
          -DBUILD_CAFFE=OFF \
          -DDOWNLOAD_BODY_COCO_MODEL=ON \
          -DDOWNLOAD_BODY_MPI_MODEL=ON \
          .. && \
    make all -j8 && \
    make install

# MTC
ENV MTC_ROOT=/opt/mtc
WORKDIR ${MTC_ROOT}
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture.git .
RUN bash ./download.sh
RUN sed -i 's/set(IGL_INCLUDE_DIRS "\/home\/donglaix\/tools\/libigl\/include\/")/set(IGL_INCLUDE_DIRS "\/opt\/libigl\/include\/")/' ${MTC_ROOT}/FitAdam/CMakeLists.txt
RUN cd FitAdam && mkdir build && cd build && \
    cmake -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -D OpenCV_DIR=/opt/opencv/build .. && \
    make all -j8

# Python Fixes
RUN pip3 install --upgrade --ignore-installed python-dateutil
RUN apt-get update
RUN apt-get install -y python3-tk
RUN pip3 install --upgrade --ignore-installed scipy

# OpenGL for X11
RUN apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module dbus-x11
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev && \
    rm -rf /var/lib/apt/lists/*
# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/lib/x86_64-linux-gnu \
  /usr/local/lib/x86_64-linux-gnu
# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json
RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Usage
# docker run --gpus 0 -it \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY -e XAUTHORITY \
#     -e NVIDIA_DRIVER_CAPABILITIES=all 
#      <docker_image_tag>
