FROM ubuntu:20.04

SHELL [ "/bin/bash", "-c" ]

# Setup user account
ARG uid=1000
ARG gid=1000
ARG USER_NAME="webdev" && USER_PASSWORD='nvidia'
RUN groupadd -r -f -g ${gid} ${USER_NAME} && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash ${USER_NAME}
RUN usermod -aG sudo ${USER_NAME}
RUN echo ${USER_NAME}:${USER_PASSWORD} | chpasswd
RUN chown ${USER_NAME} /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Repair the GPG signing key rotation for CUDA repositories
# https://github.com/NVIDIA/cuda-repo-management/issues/4
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    git-lfs \
    pkg-config \
    sudo \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    nodejs

# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&
    cd /usr/local/bin &&
    ln -s /usr/bin/python3 python &&
    ln -s /usr/bin/pip3 pip
# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools >=41.0.0
RUN pip3 config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

WORKDIR /workspace
ENTRYPOINT [ "/bin/bash" ]
