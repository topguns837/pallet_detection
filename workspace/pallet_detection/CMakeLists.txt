cmake_minimum_required(VERSION 3.8)
project(pallet_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/pallet_detection_node.py
  scripts/pallet_segmentation_node.py
  scripts/image_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install models files
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
