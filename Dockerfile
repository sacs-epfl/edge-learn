FROM python:3.9

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Move files to container
COPY src/edge_learn/requirements.txt /edge_learn/requirements.txt

# Install dependencies
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install -r /edge_learn/requirements.txt

# Install Apex from the source
RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Add edge_learn & decentralizepy to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/"

WORKDIR /edge_learn/