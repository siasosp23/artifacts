#FROM nvcr.io/nvidia/pytorch:20.10-py3
FROM docker.pdl.cmu.edu/pytorch-ib:1.9.0-cuda11.1-cudnn8-devel
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY benchmark/models/imagenet/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
COPY adaptdl/adaptdl/torch/profiles /profiles
COPY adaptdl/adaptdl/torch/seed /seed
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/imagenet imagenet
ENV PYTHONPATH=/root/imagenet:$PYTHONPATH
WORKDIR /root/imagenet

RUN cd /root/imagenet/deps/apex && pip3 install -v --disable-pip-version-check --no-cache-dir ./

ENV PYTHONUNBUFFERED=true
