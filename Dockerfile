# Use cuda_version arg to take CUDA version as input from user
ARG cuda_version=11.8.0

# Use NVIDA-CUDA's base image
FROM nvcr.io/nvidia/cuda:${cuda_version}-devel-ubuntu22.04

ARG ROS_DISTRO=humble

# Prevent console from interacting with the user
ARG DEBIAN_FRONTEND=noninteractive

# Prevent hash mismatch error for apt-get update, qqq makes the terminal quiet while downloading pkgs
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -yqqq

# Set folder for RUNTIME_DIR. Only to prevent warnings when running RViz2 and Gz
RUN mkdir tmp/runtime-root && chmod 0700 tmp/runtime-root
ENV XDG_RUNTIME_DIR='/tmp/runtime-root'

RUN apt-get update

RUN apt-get install --no-install-recommends -yqqq \
    apt-utils \
    nano \
    git 

# Using shell to use bash commands like 'source'
SHELL ["/bin/bash", "-c"]

# Python Dependencies
RUN apt-get install --no-install-recommends -yqqq \
    python3-pip

# Add locale
RUN locale  && \
    apt update && apt install --no-install-recommends -yqqq locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    locale  

# Setup the sources
RUN apt-get update && apt-get install --no-install-recommends -yqqq software-properties-common curl && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/nul

# Install ROS 2 Humble
RUN apt update && apt install --no-install-recommends -yqqq \
    ros-$ROS_DISTRO-ros-base \
    ros-dev-tools

# ROS Dependencies
RUN apt-get install --no-install-recommends -y \
    ros-$ROS_DISTRO-cyclonedds \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-cv-bridge

# Target workspace for ROS2 packages
ARG WORKSPACE=/root/pallet_ws

# Add target workspace in environment
ENV WORKSPACE=$WORKSPACE

# Create folders
RUN mkdir -p $WORKSPACE/src

# Install Python dependencies
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

# Copy relevant folders and files
COPY workspace $WORKSPACE/src/
COPY docker_scripts/run_scripts/ /root/run_scripts

# Using shell to use bash commands like 'source'
SHELL ["/bin/bash", "-c"]

# Build the ROS 2 Package
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd $WORKSPACE && \
    colcon build --symlink-install

# Make /root/pallet_ws your default directory
WORKDIR $WORKSPACE

# Source ROS 2 Humble and workspace
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc && \
    echo "source $WORKSPACE/install/setup.bash" >> /root/.bashrc
