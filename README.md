## About

The purpose of this package is to perform segmentation and detection of pallets from a given image/camera feed
<br>

## Dependencies

- [Docker](https://docs.docker.com/engine/install/)
<br>

## Installation

### Create a new workspace

```bash
mkdir -p pallet_ws/src
```
### Clone the above project inside src

```bash
cd pallet_ws/src
git clone git@github.com:topguns837/pallet_detection.git
```

### Build Docker Image 
        
Use these commands to build the Docker image. The default CUDA version is `11.8`
```bash
cd pallet_detection
docker build --build-arg cuda_version=11.8.0 -t pallet_detection:latest .
```
<br>


## Run Instructions

To run the pallet object detection model, execute the `start_pallet_detection.sh` file as shown below :

```bash
cd docker_scripts/bash_scripts
./start_pallet_detection.sh
```

To run the pallet semantic segmentation model, execute the `start_pallet_segmentation.sh` file as shown below :

```bash
cd docker_scripts/bash_scripts
./start_pallet_segmentation.sh
```

Alternatively, to run the image_publisher script for testing follow these commands :

```bash
docker exec -it pallet_detection bash
```
This will open a bash session inside the docker container, now run the script using :

```bash
ros2 run pallet_detection image_publisher.py
```

<br>

## ROS 2 API's

- Subscribed Topics 

    - `/zed2i/zed_node/rgb/image_rect_color`([sensor_msgs/msg/Image](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)) : 
            It is used to receive the RGB feed from the ZED2i camera

    - `/zed2i/zed_node/depth/depth_registered`([sensor_msgs/msg/Image](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)) : 
            It is used to get the Depth feed from the ZED2i camera
<br>


## Link to Notebooks

- [Pallet Detection using YOLOv8](https://colab.research.google.com/drive/1PuOnagpb5IB4oiAFp1_wxQlE78O-GwnD?usp=sharing)
- [Pallet Segmentation using SegForm](https://colab.research.google.com/drive/1Of36gTsnxTC9qJnbSqNIBXfn9_QPPjQK?usp=sharing)

## Pallet Detection Demo

<img src="res/pallet_detection.gif" width=700>

## Pallet Segmentation Demo

<img src="res/pallet_segmentation.gif" width=700>
