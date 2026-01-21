# DepthAI v3 with ROS2 Humble
# Using ros:humble base since depthai-ros image has old v2 depthai
FROM ros:humble
SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY ./datatypes ros2_ws/datatypes
COPY ./camera ros2_ws/camera

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip libopencv-dev libusb-1.0-0-dev

# DepthAI v3 for Model Hub and new API
# depthai-nodes provides parsing for complex models (pose, segmentation, etc.)
RUN pip install "depthai>=3.0.0" depthai-nodes opencv-python numpy

RUN cd ros2_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build

COPY ./ros_entrypoint.sh /

RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]