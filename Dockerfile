# Use an NVIDIA CUDA base image with compatible Python version
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set non-interactive installation mode for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Python check
RUN python3 --version

# Install pip3 for Python 3.9
RUN apt-get install -y python3-pip

# Move files to container
COPY src/edge_learn/requirements.txt /edge_learn/requirements.txt

# Install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install -r /edge_learn/requirements.txt

# Install git
RUN apt-get install -y git

# Install Apex from the source
RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Add edge_learn & decentralizepy to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/"

WORKDIR /edge_learn/