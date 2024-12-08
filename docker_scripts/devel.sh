#!/bin/bash

# http://wiki.ros.org/docker/Tutorials/GUI
xhost +local:root

./stop.sh

docker run -t -d --privileged --net=host \
--name pallet_detection \
-v $PWD/../workspace/:/root/pallet_ws/src/ \
-v $PWD/ddsconfig.xml:/ddsconfig.xml \
--env CYCLONEDDS_URI=/ddsconfig.xml \
--env="QT_X11_NO_MITSHM=1"  \
--env="DISPLAY"  \
pallet_detection:latest
