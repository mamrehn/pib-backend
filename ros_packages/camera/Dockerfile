FROM ros:jazzy-ros-core AS builder

SHELL ["/bin/bash", "-c"]

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-colcon-common-extensions \
    python3-pip \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./datatypes ros2_ws/datatypes
COPY ./camera ros2_ws/camera

RUN cd ros2_ws && \
    source /opt/ros/jazzy/setup.bash && \
    colcon build

# --- runtime stage (minimal) ---
FROM ros:jazzy-ros-core

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PEP 668: Jazzy (Ubuntu 24.04, Python 3.12) requires --break-system-packages
# --ignore-installed: avoid conflict with system numpy (installed by ros:jazzy-ros-core)
# opencv-python-headless pinned to 4.x: the latest 5.x release removed
# cv2.CascadeClassifier from the core module (stereo.py's face detection
# needs it), causing "module 'cv2' has no attribute 'CascadeClassifier'".
# depthai pinned to 3.6.1, the newest release whose device firmware can start
# the OAK-D Lite's OV7251 mono sensors. From 3.7.1 on (checked 3.7.1, 3.8.0,
# 3.9.0 and 3.10.0) the firmware asserts in PlgSrcMipi ("Start Source: Invalid
# config steps") as soon as a mono stream starts, and takes the colour stream
# down with it. Reproduced on two separate units on the robot; the same units
# stream both monos and stereo under 3.3.0, 3.5.0, 3.6.1 and under v2. Do not
# raise this pin without rerunning scripts/oak_bringup_check.py on hardware.
# v3 Node API migration: Jira PR-1484.
# depthai-nodes supplies the parsers for Model Hub networks whose outputs are
# not plain detections (keypoints, lines, gaze). 0.5.2 is the newest release
# that accepts depthai 3.6.1 (0.6.0 needs >=3.8.0).
RUN pip install --break-system-packages --no-cache-dir --ignore-installed \
        depthai==3.6.1 \
        depthai-nodes==0.5.2 \
        "opencv-python-headless<5"

# stereo.py needs cv2.data.haarcascades at runtime for face detection
# (CameraNode raises RuntimeError and permanently falls back to an
# error-only publisher if the cascade XML is missing). Neither the
# opencv-python nor opencv-python-headless PyPI wheels ship the haarcascade
# data files on this arm64/Python 3.12 build. The XML is vendored in the repo
# (assets/) and COPYed in, so the image builds offline / on a fresh Raspi
# without depending on network access to GitHub at build time.
COPY ./camera/assets/haarcascade_frontalface_default.xml \
     /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml

WORKDIR /app

COPY --from=builder /app/ros2_ws/install /app/ros2_ws/install

COPY ./ros_entrypoint.sh /

RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
