#FROM nvcr.io/nvidia/pytorch:20.10-py3
FROM docker.pdl.cmu.edu/pytorch-ib:1.9.0-cuda11.1-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list && apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY benchmark/models/deepspeech2/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install libsndfile1 -y

COPY adaptdl adaptdl
COPY adaptdl/adaptdl/torch/profiles /profiles
COPY adaptdl/adaptdl/torch/seed /seed
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/deepspeech2 deepspeech2
ENV PYTHONPATH=/root/depspeech2:$PYTHONPATH
WORKDIR /root/deepspeech2

ENV PYTHONUNBUFFERED=true
