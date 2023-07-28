#FROM nvcr.io/nvidia/pytorch:20.03-py3
FROM docker.pdl.cmu.edu/pytorch-ib:1.9.0-cuda11.1-cudnn8-devel
WORKDIR /root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
COPY adaptdl/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY benchmark/models/cifar10/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
COPY adaptdl/adaptdl/torch/profiles /profiles
COPY adaptdl/adaptdl/torch/seed /seed
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/cifar10 cifar10
ENV PYTHONPATH=/root/cifar10:$PYTHONPATH
WORKDIR /root/cifar10

ENV PYTHONUNBUFFERED=true
