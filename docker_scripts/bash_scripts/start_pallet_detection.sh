#!/bin/bash

xhost +local:root

./stop.sh

docker run -t -d --privileged --net=host \
--name pallet_detection \
-v $PWD/ddsconfig.xml:/ddsconfig.xml \
--env CYCLONEDDS_URI=/ddsconfig.xml \
--env="QT_X11_NO_MITSHM=1"  \
--env="DISPLAY"  \
pallet_detection:latest

docker exec -it pallet_detection /root/run_scripts/run_pallet_detection.sh

xhost -local:root