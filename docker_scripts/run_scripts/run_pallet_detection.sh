#!/bin/bash

source /opt/ros/humble/setup.bash

source $WORKSPACE/install/setup.bash

ros2 run pallet_detection pallet_detection_node.py
