#FROM nvcr.io/nvidia/pytorch:20.10-py3
FROM docker.pdl.cmu.edu/pytorch-ib:1.9.0-cuda11.1-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list && apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt

RUN apt update
RUN apt install -f
RUN df -h
RUN lsblk
RUN apt-get autoclean && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install --no-cache-dir -r requirements.txt
COPY benchmark/models/yolov3/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
COPY adaptdl/adaptdl/torch/profiles /profiles
COPY adaptdl/adaptdl/torch/seed /seed
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/yolov3 yolov3
ENV PYTHONPATH=/root/yolov3:$PYTHONPATH
WORKDIR /root/yolov3

RUN cd /root/yolov3/deps/apex && pip3 install -v --disable-pip-version-check --no-cache-dir ./ 
#--global-option="--cpp_ext" --global-option="--cuda_ext" ./

ENV PYTHONUNBUFFERED=true
