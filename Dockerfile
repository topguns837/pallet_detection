# ROS2 humble Base
FROM ros:humble

ARG ROS_DISTRO=humble

# Prevent hash mismatch error for apt-get update
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -y

# Non Python/ROS Dependencies
RUN apt-get install --no-install-recommends -y \
    vim \
    software-properties-common

# Python Dependencies
RUN apt-get install --no-install-recommends -y \
    python3-pip

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

COPY requirements.txt /root

RUN pip install -r /root/requirements.txt

COPY workspace $WORKSPACE/src/

COPY docker_scripts/run_scripts/ /root/run_scripts

# Using shell to use bash commands like 'source'
SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd $WORKSPACE && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install

WORKDIR $WORKSPACE

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc && \
    echo "source $WORKSPACE/install/setup.bash" >> /root/.bashrc
