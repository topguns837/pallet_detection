#!/bin/bash

xhost +local:root

./stop.sh

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=6710886 \
-t -d --privileged --net=host \
--name pallet_detection \
-v $PWD/ddsconfig.xml:/ddsconfig.xml \
--env CYCLONEDDS_URI=/ddsconfig.xml \
--env="QT_X11_NO_MITSHM=1"  \
--env="DISPLAY"  \
pallet_detection:latest

docker exec -it pallet_detection /root/run_scripts/run_pallet_segmentation.sh

xhost -local:root
