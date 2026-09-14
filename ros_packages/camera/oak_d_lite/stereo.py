#!/usr/bin/python3
"""
OAK-D Lite camera node (DepthAI v3).

Combines the upstream camera node with the ai_cam_topics feature set:

Always-on (upstream behaviour, the services read these caches):
- Colour ISP stream -> ``camera_topic`` (base64 JPEG), ``camera/rgb/image``
  (same payload) and ``camera/image/compressed`` (binary CompressedImage).
- Haar-cascade face tracking -> ``face_center``, skipped when unsubscribed.
- StereoDepth -> ``stereo_depth`` plus the ``get_depth_frame`` and
  ``get_distance_at_px`` services.

On demand (the pipeline is rebuilt only when subscriber demand changes):
- AI inference via the Luxonis Model Hub -> ``camera/ai/*``.
- BMI270 IMU streaming -> ``camera/imu*``.

The OAK-D Lite allows a single running pipeline, so colour, depth, AI and IMU
all live in one ``dai.Pipeline`` that is started exactly once per build.
"""

import base64
import json
import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import depthai as dai
import numpy as np
import rclpy
from datatypes.srv import GetCameraImage, GetDepthFrame, GetDistanceAtPx, SwitchModel
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Float32MultiArray, Float64, Int32, Int32MultiArray, String

# Downscaled resolution for Haar cascade face detection (maps back to full frame).
FACE_DETECT_WIDTH = 320
FACE_DETECT_HEIGHT = 180

# ============== MODEL REGISTRY (DepthAI v3 Model Hub) ==============
# Format: "luxonis/model-name:variant" for Model Hub.
# All models are optimized for RVC2 (OAK-D Lite).
# Slugs must match scripts/download_oak_models.py for pre-caching.

AVAILABLE_MODELS = {
    # ============== OBJECT DETECTION ==============
    "yolov6n": {
        "type": "detection",
        "slug": "luxonis/yolov6-nano:r2-coco-512x288",
        "description": "YOLOv6 Nano - fast & accurate object detection",
        "classes": 80,  # COCO classes
        "node_type": "DetectionNetwork",
    },
    "yolov10n": {
        "type": "detection",
        "slug": "luxonis/yolov10-nano:coco-512x288",
        "description": "YOLOv10 Nano - latest YOLO architecture",
        "classes": 80,
        "node_type": "DetectionNetwork",
    },
    "person": {
        "type": "detection",
        "slug": "luxonis/scrfd-person-detection:25g-640x640",
        "description": "SCRFD Person detector - optimized for people detection",
        "classes": 1,
        "node_type": "DetectionNetwork",
    },
    "face": {
        "type": "detection",
        "slug": "luxonis/yunet:640x480",
        "description": "YuNet face detection - fast and reliable",
        "classes": 1,
        "node_type": "DetectionNetwork",
    },
    # ============== POSE ESTIMATION ==============
    "pose_yolo": {
        "type": "pose",
        "slug": "luxonis/yolov8-nano-pose-estimation:coco-512x288",
        "description": "YOLOv8 Pose - 17 keypoint body pose",
        "keypoints": 17,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "ImgDetectionsExtended",
    },
    "pose_hrnet": {
        "type": "pose",
        "slug": "luxonis/lite-hrnet:18-coco-288x384",
        "description": "Lite-HRNet - high resolution pose estimation",
        "keypoints": 17,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Keypoints",
    },
    # ============== HAND DETECTION ==============
    "hand": {
        "type": "hand",
        "slug": "luxonis/mediapipe-hand-landmarker:224x224",
        "description": "MediaPipe hand landmark detection",
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Keypoints",
    },
    # ============== SEGMENTATION ==============
    "segmentation": {
        "type": "instance-segmentation",
        "slug": "luxonis/yolov8-instance-segmentation-nano:coco-512x288",
        "description": "YOLOv8 Instance Segmentation",
        "classes": 80,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "ImgDetectionsExtended",
    },
    # ============== GAZE ESTIMATION ==============
    "gaze": {
        "type": "gaze",
        "slug": "luxonis/l2cs-net:448x448",
        "description": "L2CS-Net gaze estimation",
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Predictions",
    },
    # ============== LINE DETECTION ==============
    "lines": {
        "type": "lines",
        "slug": "luxonis/m-lsd:512x512",
        "description": "M-LSD line segment detection",
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Lines",
    },
}

DEFAULT_MODEL = "yolov6n"

# Valid IMU frequencies for the BMI270 sensor.
BMI270_VALID_FREQUENCIES = [25, 50, 100, 200, 400]

# Colour and depth are always part of the pipeline; only these are negotiated.
DEFAULT_PIPELINE_CONFIG = {"ai": False, "imu": False}


def rle_encode(mask: np.ndarray) -> Dict[str, Any]:
    """
    RLE-encode a binary mask for efficient transmission.
    Much smaller than raw mask data.
    """
    flat = mask.flatten().astype(np.uint8)
    runs = []
    values = []

    if len(flat) == 0:
        return {"runs": [], "values": [], "shape": list(mask.shape)}

    current_val = flat[0]
    run_length = 1

    for i in range(1, len(flat)):
        if flat[i] == current_val:
            run_length += 1
        else:
            runs.append(run_length)
            values.append(int(current_val))
            current_val = flat[i]
            run_length = 1

    runs.append(run_length)
    values.append(int(current_val))

    return {"runs": runs, "values": values, "shape": list(mask.shape)}


class ErrorPublisher(Node):
    """Fallback node that reports the camera as unavailable."""

    def __init__(self):
        super().__init__("error_publisher")
        self.publisher_ = self.create_publisher(String, "camera_topic", 10)
        self.error_pub = self.create_publisher(String, "camera/error", 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.current_image = ""

    def timer_callback(self):
        msg = String()
        msg.data = "Camera not available: "
        self.publisher_.publish(msg)

        error_msg = String()
        error_msg.data = json.dumps(
            {
                "error": "OAK-D Lite camera failed to initialize",
                "timestamp": time.time(),
            }
        )
        self.error_pub.publish(error_msg)


class CameraNode(Node):
    """OAK-D Lite node serving colour, depth, face tracking, AI and IMU."""

    def __init__(self):
        super().__init__("camera_node")

        self._check_depthai_version()

        # ============== PUBLISHERS ==============
        # Colour preview - legacy base64 (Cerebra, Blockly, docs/test-basis).
        self.publisher_ = self.create_publisher(String, "camera_topic", 10)
        # Namespaced alias carrying the identical base64 payload.
        self.rgb_pub = self.create_publisher(String, "camera/rgb/image", 10)
        # Binary JPEG - avoids the ~33% base64 overhead for new consumers.
        self.camera_image_pub = self.create_publisher(
            CompressedImage, "camera/image/compressed", 10
        )
        self.depth_publisher_ = self.create_publisher(String, "stereo_depth", 10)
        self.face_center_publisher_ = self.create_publisher(
            Float32MultiArray, "face_center", 10
        )

        # AI
        self.ai_pub = self.create_publisher(String, "camera/ai/detections", 10)
        self.ai_current_pub = self.create_publisher(
            String, "camera/ai/current_model", 10
        )
        self.ai_available_pub = self.create_publisher(
            String, "camera/ai/available_models", 10
        )
        self.ai_status_pub = self.create_publisher(String, "camera/ai/status", 10)

        # IMU
        self.imu_pub = self.create_publisher(Imu, "camera/imu", 50)
        self.imu_accel_pub = self.create_publisher(
            Vector3Stamped, "camera/imu/accelerometer", 50
        )
        self.imu_gyro_pub = self.create_publisher(
            Vector3Stamped, "camera/imu/gyroscope", 50
        )

        self.error_pub = self.create_publisher(String, "camera/error", 10)

        cascade_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]

        cascade_path = next((p for p in cascade_paths if os.path.exists(p)), None)

        if cascade_path is None:
            raise RuntimeError("haarcascade_frontalface_default.xml not found")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # ============== SUBSCRIBERS ==============
        # Legacy control topics.
        self.timer_subscription = self.create_subscription(
            Float64, "timer_period_topic", self.timer_period_callback, 10
        )
        self.quality_factor_subscription = self.create_subscription(
            Int32, "quality_factor_topic", self.quality_factor_callback, 10
        )
        self.preview_size_subscription = self.create_subscription(
            Int32MultiArray, "size_topic", self.preview_size_callback, 10
        )
        # Namespaced aliases for the same controls.
        self.create_subscription(
            Float64, "camera/timer_period", self.timer_period_callback, 10
        )
        self.create_subscription(
            Int32, "camera/quality_factor", self.quality_factor_callback, 10
        )
        self.create_subscription(
            Int32MultiArray, "camera/preview_size", self.preview_size_callback, 10
        )
        # Feature configuration.
        self.create_subscription(
            String, "camera/ai/config", self.ai_config_callback, 10
        )
        self.create_subscription(
            String, "camera/video/config", self.camera_config_callback, 10
        )
        self.create_subscription(
            String, "camera/imu/config", self.imu_config_callback, 10
        )

        self.preview_width = 1280
        self.preview_height = 720
        self.quality_factor = 80
        self.current_image = ""
        self.current_frame = None
        self.current_depth = None
        self.pipeline = None
        self.queue = None
        self.depth_queue = None
        self.queues: Dict[str, Any] = {}

        # Pipeline / feature state.
        self.pipeline_config = dict(DEFAULT_PIPELINE_CONFIG)
        self._force_rebuild = False
        self._pipeline_lock = threading.RLock()

        # Model state.
        self.current_model_name = DEFAULT_MODEL
        self._model_loading = False
        self._model_load_error: Optional[str] = None
        # A model that fails to build must not be retried on every tick: that
        # would tear down colour and depth ~10x a second. Remember the model
        # that failed and stop asking for it until the selection changes.
        self._ai_failed_model: Optional[str] = None
        self._imu_unavailable = False

        # AI settings.
        self.ai_confidence = 0.5
        self.segmentation_mode = "bbox"  # "bbox" or "mask"
        self.segmentation_target_class = None  # None = all classes
        self._frame_count = 0

        # IMU settings.
        self.imu_freq = 100
        self.imu_actual_freq = self._validate_imu_frequency(100)

        self.camera_available = self.init_pipeline()

        if self.camera_available:
            self.get_camera_image_service = self.create_service(
                GetCameraImage, "get_camera_image", self.get_camera_image_callback
            )
            self.get_depth_frame_service = self.create_service(
                GetDepthFrame, "get_depth_frame", self.get_depth_frame_callback
            )
            self.get_distance_at_px_service = self.create_service(
                GetDistanceAtPx, "get_distance_at_px", self.get_distance_at_px_callback
            )
            self.switch_ai_model_service = self.create_service(
                SwitchModel, "switch_ai_model", self.switch_model_callback
            )
            self.get_logger().info("Camera service initialized.")
        else:
            self.get_logger().error("Camera not available.")

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.status_timer = self.create_timer(1.0, self.status_timer_callback)

        self._publish_status("idle", "Ready - waiting for subscribers")

    # ============== HELPERS ==============

    def _check_depthai_version(self):
        """Warn when running against a DepthAI older than v3 (the v3 API is used)."""
        try:
            version = getattr(dai, "__version__", "0.0.0")
            major = int(str(version).split(".")[0])
        except (TypeError, ValueError):
            return
        if major < 3:
            self.get_logger().error(
                f"DepthAI v{version} detected. This node requires DepthAI v3.0.0+. "
                "Upgrade with: pip install 'depthai>=3.0.0'"
            )
        else:
            self.get_logger().info(f"DepthAI version: {version}")

    def _validate_imu_frequency(self, requested: int) -> int:
        """Find the closest valid BMI270 frequency."""
        return min(BMI270_VALID_FREQUENCIES, key=lambda x: abs(x - requested))

    def _publish_status(self, state: str, message: str = "", model: str = ""):
        """Publish an AI status update to camera/ai/status."""
        if not model:
            model = self.current_model_name
        status = {
            "state": state,  # "idle", "loading", "ready", "error"
            "model": model,
            "message": message,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.ai_status_pub.publish(msg)

    def _encode_frame(self, frame):
        """JPEG-encode and base64 a frame; returns None on failure."""
        retval, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_factor]
        )
        if not retval:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def _encode_depth_frame(self, depth):
        """Pack uint16 depth (mm) as metadata + base64; returns None on failure."""
        if depth is None:
            return None
        try:
            depth_u16 = np.ascontiguousarray(depth, dtype=np.uint16)
            height, width = depth_u16.shape[:2]
            encoded = base64.b64encode(depth_u16.tobytes()).decode("utf-8")
            return width, height, "16UC1", encoded
        except Exception as e:
            self.get_logger().error(f"Failed to encode depth frame: {e}")
            return None

    def _get_model_description(self, model_name: str):
        """Get the NNModelDescription for a model from the registry."""
        model_info = AVAILABLE_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")

        slug = model_info.get("slug")
        if not slug:
            raise ValueError(f"Model {model_name} has no slug defined")

        self.get_logger().info(f"Creating model description for: {slug}")
        model_desc = dai.NNModelDescription(slug)
        # Set platform for RVC2 (OAK-D Lite).
        model_desc.platform = "RVC2"
        return model_desc

    # ============== SERVICE CALLBACKS ==============

    def get_camera_image_callback(self, request, response):
        # Encode on demand when the service is requested and we have a cached frame.
        if self.current_frame is not None:
            encoded = self._encode_frame(self.current_frame)
            if encoded is not None:
                self.current_image = encoded
        self.get_logger().info(f"LEN IMAGE: {len(self.current_image)}")
        response.image_base64 = self.current_image
        return response

    def get_depth_frame_callback(self, request, response):
        # Read cached depth from the persistent pipeline; do not reconnect.
        packed = self._encode_depth_frame(self.current_depth)
        if packed is None:
            response.width = 0
            response.height = 0
            response.encoding = ""
            response.depth_base64 = ""
            return response
        response.width, response.height, response.encoding, response.depth_base64 = (
            packed
        )
        return response

    def get_distance_at_px_callback(self, request, response):
        # Pixel lookup against cached depth (mm). 0 means invalid / out of range.
        response.distance_mm = 0.0
        if self.current_depth is None:
            return response
        height, width = self.current_depth.shape[:2]
        x, y = int(request.x), int(request.y)
        if x < 0 or y < 0 or x >= width or y >= height:
            return response
        response.distance_mm = float(self.current_depth[y, x])
        return response

    def switch_model_callback(self, request, response):
        """Service to switch the active AI model."""
        model_name = request.model_name

        if model_name not in AVAILABLE_MODELS:
            response.success = False
            response.message = (
                f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS)}"
            )
            return response

        if model_name == self.current_model_name:
            response.success = True
            response.message = f"Already using {model_name}"
            return response

        self.get_logger().info(
            f"Switching model: {self.current_model_name} -> {model_name}"
        )
        self.current_model_name = model_name
        self._model_load_error = None
        self._ai_failed_model = None
        self._publish_status("loading", f"Switching to model {model_name}...")

        if self.pipeline_config.get("ai"):
            self._force_rebuild = True
            self.check_demand()

        if self._model_load_error:
            response.success = False
            response.message = f"Model switch failed: {self._model_load_error}"
            return response

        response.success = True
        response.message = (
            f"Switched to {model_name}. "
            "Subscribe to camera/ai/status for loading progress."
        )
        return response

    # ============== PIPELINE ==============

    def _init_stereo_depth(self):
        """Add StereoDepth outputs to the existing pipeline (no extra start)."""
        mono_left = self.pipeline.create(dai.node.Camera)
        mono_left.build(dai.CameraBoardSocket.CAM_B)
        mono_right = self.pipeline.create(dai.node.Camera)
        mono_right.build(dai.CameraBoardSocket.CAM_C)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        except Exception:
            pass
        stereo.setLeftRightCheck(True)
        try:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        except Exception:
            self.get_logger().warning(
                "Depth-to-RGB align unavailable; using native depth."
            )

        mono_left_out = mono_left.requestFullResolutionOutput()
        mono_right_out = mono_right.requestFullResolutionOutput()
        mono_left_out.link(stereo.left)
        mono_right_out.link(stereo.right)

        self.depth_queue = stereo.depth.createOutputQueue()
        self.get_logger().info("Stereo depth outputs added to pipeline.")

    def _init_ai(self, camera_node):
        """Attach the currently selected model to the pipeline."""
        model_name = self.current_model_name
        model_info = AVAILABLE_MODELS.get(model_name, {})
        node_type = model_info.get("node_type", "DetectionNetwork")

        self.get_logger().info(
            f"Loading model: {model_name} (type={model_info.get('type')}, "
            f"node={node_type}) - this may take a moment if downloading..."
        )

        model_desc = self._get_model_description(model_name)

        if node_type == "DetectionNetwork":
            # v3 API: DetectionNetwork.build(cameraNode, modelDesc)
            nn = self.pipeline.create(dai.node.DetectionNetwork).build(
                camera_node, model_desc
            )
            self.queues["nn"] = nn.out.createOutputQueue(maxSize=4, blocking=False)
            # Passthrough for getting frames alongside detections.
            self.queues["nn_passthrough"] = nn.passthrough.createOutputQueue(
                maxSize=4, blocking=False
            )
            return

        if node_type == "ParsingNeuralNetwork":
            # Complex models with custom parsers (depthai-nodes).
            try:
                from depthai_nodes import ParsingNeuralNetwork

                nn = ParsingNeuralNetwork.build(camera_node, model_desc)
                self.queues["nn"] = nn.out.createOutputQueue(maxSize=4, blocking=False)
                return
            except ImportError:
                self.get_logger().error(
                    "depthai-nodes not installed. "
                    "Install with: pip install depthai-nodes"
                )

        # Generic NeuralNetwork fallback (also used when depthai-nodes is missing).
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setNNModelDescription(model_desc)
        nn.input.setBlocking(False)
        camera_node.requestOutput(
            (self.preview_width, self.preview_height),
            type=dai.ImgFrame.Type.BGR888p,
        ).link(nn.input)
        self.queues["nn"] = nn.out.createOutputQueue(maxSize=4, blocking=False)

    def _init_imu(self):
        """Attach the BMI270 IMU to the pipeline."""
        imu = self.pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, self.imu_actual_freq)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, self.imu_actual_freq)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        self.queues["imu"] = imu.out.createOutputQueue(maxSize=50, blocking=False)

    def init_pipeline(self) -> bool:
        """
        Build and start the single device pipeline.

        Colour ISP and StereoDepth are always present; AI and IMU are added when
        ``self.pipeline_config`` asks for them. ``pipeline.start()`` is called
        exactly once (OAK-D-Lite 1-pipeline constraint).
        """
        config = dict(getattr(self, "pipeline_config", DEFAULT_PIPELINE_CONFIG))
        self.queues = {}
        try:
            # depthai v3 API:
            # Camera node replaces deprecated ColorCamera.
            # XLinkOut is completely removed in DepthAI v3.
            # Output queues are created directly from node outputs.
            # Colour ISP, StereoDepth, the optional neural network and the
            # optional IMU all share this single pipeline.
            self.pipeline = dai.Pipeline()
            self.camRgb = self.pipeline.create(dai.node.Camera)
            self.camRgb.build(dai.CameraBoardSocket.CAM_A)
            self.isp_out = self.camRgb.requestIspOutput()

            self.queue = self.isp_out.createOutputQueue()
            self.depth_queue = None

            try:
                self._init_stereo_depth()
            except Exception as stereo_exc:
                self.get_logger().warning(f"Stereo depth not available: {stereo_exc}")
                self.depth_queue = None

            if config.get("ai"):
                self._model_loading = True
                try:
                    self._init_ai(self.camRgb)
                    self._model_load_error = None
                    self._ai_failed_model = None
                except Exception as ai_exc:
                    # AI is optional: keep colour/depth/IMU running without it.
                    self.get_logger().error(f"AI model not available: {ai_exc}")
                    self._model_load_error = str(ai_exc)
                    self._ai_failed_model = self.current_model_name
                    config["ai"] = False
                    self.queues.pop("nn", None)
                    self.queues.pop("nn_passthrough", None)
                finally:
                    self._model_loading = False

            if config.get("imu"):
                try:
                    self._init_imu()
                except Exception as imu_exc:
                    self.get_logger().warning(f"IMU not available: {imu_exc}")
                    self._imu_unavailable = True
                    config["imu"] = False
                    self.queues.pop("imu", None)

            self.pipeline.start()
            self.pipeline_config = config

            self.get_logger().info("DepthAI v3 pipeline started successfully.")
            return True

        except Exception as e:
            import traceback

            print("====================================")
            print("CAMERA INIT FAILED")
            traceback.print_exc()
            print("====================================")

            self.get_logger().error(f"Camera not found: {e}")
            self.pipeline = None
            self.queue = None
            self.depth_queue = None
            self.queues = {}
            self.pipeline_config = dict(DEFAULT_PIPELINE_CONFIG)
            return False

    def _restart_pipeline(self, config) -> bool:
        """Stop the running pipeline and rebuild it with ``config``."""
        with self._pipeline_lock:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    self.get_logger().warning(f"Error stopping pipeline: {e}")
                self.pipeline = None

            self.pipeline_config = dict(config)
            ok = self.init_pipeline()

            if not ok:
                self._publish_status("error", "Pipeline restart failed")
            elif self.pipeline_config.get("ai"):
                self._publish_status(
                    "ready", f"Model {self.current_model_name} loaded successfully"
                )
            elif self._model_load_error:
                self._publish_status("error", self._model_load_error)
            else:
                self._publish_status("idle", "AI inactive - no subscribers")
            return ok

    def check_demand(self):
        """Enable/disable the on-demand AI and IMU branches of the pipeline."""
        need_ai = (
            self.ai_pub.get_subscription_count() > 0
            and self.current_model_name != self._ai_failed_model
        )
        need_imu = (
            self.imu_pub.get_subscription_count() > 0
            or self.imu_accel_pub.get_subscription_count() > 0
            or self.imu_gyro_pub.get_subscription_count() > 0
        ) and not self._imu_unavailable
        new_config = {"ai": need_ai, "imu": need_imu}

        changed = new_config != self.pipeline_config
        if self._force_rebuild:
            changed = True
            self._force_rebuild = False

        if not changed:
            return

        self.get_logger().info(f"Demand changed: {new_config}. Rebuilding pipeline...")
        if need_ai:
            self._publish_status(
                "loading", f"Loading model {self.current_model_name}..."
            )
        self._restart_pipeline(new_config)

    # ============== PROCESSING ==============

    def publish_face_center(self, frame):
        # Skip expensive Haar cascade when nobody is listening to face_center.
        if self.face_center_publisher_.get_subscription_count() == 0:
            return

        face_msg = Float32MultiArray()

        if self.face_cascade.empty():
            face_msg.data = [0.0, 0.0]
            self.face_center_publisher_.publish(face_msg)
            return

        full_h, full_w = frame.shape[:2]
        small = cv2.resize(frame, (FACE_DETECT_WIDTH, FACE_DETECT_HEIGHT))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            scale_x = full_w / FACE_DETECT_WIDTH
            scale_y = full_h / FACE_DETECT_HEIGHT
            x_full = x * scale_x
            y_full = y * scale_y
            w_full = w * scale_x
            h_full = h * scale_y
            x_center = (x_full + w_full / 2) - (full_w / 2)
            y_center = (full_h / 2) - (y_full + h_full / 2)
            face_msg.data = [float(x_center), float(y_center)]
        else:
            face_msg.data = [0.0, 0.0]

        self.face_center_publisher_.publish(face_msg)

    def _publish_colorized_depth(self, depth):
        if self.depth_publisher_.get_subscription_count() == 0:
            return
        depth_u16 = np.ascontiguousarray(depth, dtype=np.uint16)
        depth_vis = cv2.normalize(depth_u16, None, 0, 255, cv2.NORM_MINMAX)
        colorized = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
        encoded = self._encode_frame(colorized)
        if encoded is None:
            return
        msg = String()
        msg.data = encoded
        self.depth_publisher_.publish(msg)

    def _publish_color_frame(self, frame):
        """JPEG-encode once and fan out to the binary and base64 colour topics."""
        want_binary = self.camera_image_pub.get_subscription_count() > 0
        want_legacy = self.publisher_.get_subscription_count() > 0
        want_rgb = self.rgb_pub.get_subscription_count() > 0

        if not (want_binary or want_legacy or want_rgb):
            return

        retval, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_factor]
        )
        if not retval:
            return

        if want_binary:
            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "oak_camera"
            img_msg.format = "jpeg"
            img_msg.data = buffer.tobytes()
            self.camera_image_pub.publish(img_msg)

        if not (want_legacy or want_rgb):
            return

        encoded = base64.b64encode(buffer).decode("utf-8")
        self.current_image = encoded
        msg = String()
        msg.data = encoded
        if want_legacy:
            self.publisher_.publish(msg)
        if want_rgb:
            self.rgb_pub.publish(msg)

    def _process_ai(self):
        q = self.queues.get("nn")
        if q is None:
            return
        while q.has():
            self._frame_count += 1

            in_data = q.get()
            model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
            model_type = model_info.get("type", "detection")
            output_type = model_info.get("output_type", "ImgDetections")

            # Calculate latency.
            try:
                frame_ts = in_data.getTimestamp()
                now = dai.Clock.now()
                latency_ms = (now - frame_ts).total_seconds() * 1000
            except Exception:
                latency_ms = 0.0

            result = self._format_ai_result(
                in_data, model_type, output_type, model_info
            )

            output = {
                "model": self.current_model_name,
                "type": model_type,
                "frame_id": self._frame_count,
                "timestamp_ns": self.get_clock().now().nanoseconds,
                "latency_ms": round(latency_ms, 2),
                "result": result,
            }

            msg = String()
            msg.data = json.dumps(output)
            self.ai_pub.publish(msg)

    def _process_imu(self):
        q = self.queues.get("imu")
        if q is None:
            return
        while q.has():
            imu_data = q.get()
            for packet in imu_data.packets:
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = "oak_imu_frame"

                accel = packet.acceleroMeter
                gyro = packet.gyroscope

                imu_msg.linear_acceleration.x = accel.x
                imu_msg.linear_acceleration.y = accel.y
                imu_msg.linear_acceleration.z = accel.z

                imu_msg.angular_velocity.x = gyro.x
                imu_msg.angular_velocity.y = gyro.y
                imu_msg.angular_velocity.z = gyro.z

                self.imu_pub.publish(imu_msg)

                accel_msg = Vector3Stamped()
                accel_msg.header = imu_msg.header
                accel_msg.vector = imu_msg.linear_acceleration
                self.imu_accel_pub.publish(accel_msg)

                gyro_msg = Vector3Stamped()
                gyro_msg.header = imu_msg.header
                gyro_msg.vector = imu_msg.angular_velocity
                self.imu_gyro_pub.publish(gyro_msg)

    def timer_callback(self):
        try:
            self.check_demand()
        except Exception as e:
            self.get_logger().error(f"Error in check_demand: {e}")

        if self.queue:
            image_rgb = self.queue.tryGet()
            if image_rgb is not None:
                frame = image_rgb.getCvFrame()

                if (
                    frame.shape[1] != self.preview_width
                    or frame.shape[0] != self.preview_height
                ):
                    frame = cv2.resize(frame, (self.preview_width, self.preview_height))

                self.current_frame = frame
                self.publish_face_center(frame)

                # Only JPEG/base64 encode when someone is subscribed to a colour
                # topic. get_camera_image encodes on demand from current_frame.
                self._publish_color_frame(frame)

        try:
            self._process_ai()
        except Exception as e:
            self.get_logger().error(f"Error processing AI: {e}")

        try:
            self._process_imu()
        except Exception as e:
            self.get_logger().error(f"Error processing IMU: {e}")

        if not self.depth_queue:
            return

        depth_packet = self.depth_queue.tryGet()
        if depth_packet is None:
            return

        depth = depth_packet.getFrame()
        self.current_depth = depth
        self._publish_colorized_depth(depth)

    # ============== AI RESULT FORMATTING ==============

    def _format_ai_result(
        self, in_data, model_type: str, output_type: str, model_info: dict
    ) -> dict:
        """Format an AI inference result based on model type and output type."""

        # Handle DepthAI v3 parsed outputs.
        if output_type == "ImgDetections" or isinstance(in_data, dai.ImgDetections):
            # Standard detections.
            return self._format_detections(in_data)

        # Try depthai-nodes output types.
        try:
            from depthai_nodes.ml.messages import (
                ImgDetectionsExtended,
                Keypoints,
                Lines,
                Predictions,
            )

            if isinstance(in_data, ImgDetectionsExtended):
                return self._format_detections_extended(in_data, model_info)
            elif isinstance(in_data, Keypoints):
                return self._format_keypoints(in_data, model_info)
            elif isinstance(in_data, Lines):
                return self._format_lines(in_data)
            elif isinstance(in_data, Predictions):
                return self._format_predictions(in_data, model_type)

        except ImportError:
            pass  # depthai-nodes not installed, fall through to raw handling.

        # Fallback: handle based on model_type for raw NeuralNetwork output.
        if model_type == "detection":
            return self._format_detections(in_data)
        elif model_type == "pose":
            return self._format_pose_raw(in_data, model_info)
        elif model_type == "instance-segmentation":
            return self._format_instance_seg_raw(in_data, model_info)
        elif model_type == "hand":
            return self._format_hand_raw(in_data, model_info)
        else:
            return {"raw": "unsupported model type", "type": model_type}

    def _format_detections(self, in_data) -> dict:
        """Format standard ImgDetections."""
        detections = []
        try:
            for det in in_data.detections:
                detections.append(
                    {
                        "label": det.label,
                        "confidence": round(det.confidence, 4),
                        "bbox": {
                            "xmin": round(det.xmin, 4),
                            "ymin": round(det.ymin, 4),
                            "xmax": round(det.xmax, 4),
                            "ymax": round(det.ymax, 4),
                        },
                    }
                )
        except Exception as e:
            return {"error": str(e)}

        return {"detections": detections, "count": len(detections)}

    def _format_detections_extended(self, in_data, model_info: dict) -> dict:
        """Format ImgDetectionsExtended (detections with keypoints/masks)."""
        detections = []
        try:
            for det in in_data.detections:
                det_dict = {
                    "label": det.label,
                    "confidence": round(det.confidence, 4),
                    "bbox": {
                        "xmin": round(det.xmin, 4),
                        "ymin": round(det.ymin, 4),
                        "xmax": round(det.xmax, 4),
                        "ymax": round(det.ymax, 4),
                    },
                }

                # Add keypoints if present.
                if hasattr(det, "keypoints") and det.keypoints:
                    det_dict["keypoints"] = []
                    for kp in det.keypoints:
                        det_dict["keypoints"].append(
                            {
                                "x": round(kp.x, 4),
                                "y": round(kp.y, 4),
                                "confidence": (
                                    round(kp.confidence, 4)
                                    if hasattr(kp, "confidence")
                                    else 1.0
                                ),
                            }
                        )

                # Add mask if present and requested.
                if hasattr(det, "mask") and det.mask is not None:
                    if self.segmentation_mode == "mask":
                        # RLE encode for efficiency.
                        mask_array = np.array(det.mask)
                        det_dict["mask_rle"] = rle_encode(mask_array)
                    # Always include a mask presence indicator.
                    det_dict["has_mask"] = True

                detections.append(det_dict)
        except Exception as e:
            return {"error": str(e)}

        return {"detections": detections, "count": len(detections)}

    def _format_keypoints(self, in_data, model_info: dict) -> dict:
        """Format Keypoints output."""
        try:
            keypoints_list = []
            for kp in in_data.keypoints:
                keypoints_list.append(
                    {
                        "x": round(kp.x, 4),
                        "y": round(kp.y, 4),
                        "confidence": (
                            round(kp.confidence, 4)
                            if hasattr(kp, "confidence")
                            else 1.0
                        ),
                    }
                )
            return {"keypoints": keypoints_list, "count": len(keypoints_list)}
        except Exception as e:
            return {"error": str(e)}

    def _format_lines(self, in_data) -> dict:
        """Format Lines output."""
        try:
            lines_list = []
            for line in in_data.lines:
                lines_list.append(
                    {
                        "start": {
                            "x": round(line.start_x, 4),
                            "y": round(line.start_y, 4),
                        },
                        "end": {"x": round(line.end_x, 4), "y": round(line.end_y, 4)},
                        "confidence": (
                            round(line.confidence, 4)
                            if hasattr(line, "confidence")
                            else 1.0
                        ),
                    }
                )
            return {"lines": lines_list, "count": len(lines_list)}
        except Exception as e:
            return {"error": str(e)}

    def _format_predictions(self, in_data, model_type: str) -> dict:
        """Format Predictions output."""
        try:
            predictions = []
            for pred in in_data.predictions:
                predictions.append(
                    {"class": pred.label, "confidence": round(pred.confidence, 4)}
                )
            return {"predictions": predictions, "count": len(predictions)}
        except Exception as e:
            return {"error": str(e)}

    def _format_pose_raw(self, in_data, model_info: dict) -> dict:
        """Format raw pose estimation output."""
        try:
            # Raw NNData handling.
            layers = in_data.getAllLayerNames()
            return {
                "raw_layers": layers,
                "note": "Install depthai-nodes for parsed output",
            }
        except Exception as e:
            return {"error": str(e)}

    def _format_instance_seg_raw(self, in_data, model_info: dict) -> dict:
        """Format raw instance segmentation output."""
        try:
            layers = in_data.getAllLayerNames()
            return {
                "raw_layers": layers,
                "note": "Install depthai-nodes for parsed output",
            }
        except Exception as e:
            return {"error": str(e)}

    def _format_hand_raw(self, in_data, model_info: dict) -> dict:
        """Format raw hand detection output."""
        try:
            if isinstance(in_data, dai.ImgDetections):
                return self._format_detections(in_data)
            layers = in_data.getAllLayerNames()
            return {
                "raw_layers": layers,
                "note": "Install depthai-nodes for parsed output",
            }
        except Exception as e:
            return {"error": str(e)}

    # ============== CONFIG CALLBACKS ==============

    def status_timer_callback(self):
        """Publish the current model and the available models."""
        model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
        msg_curr = String()
        msg_curr.data = json.dumps(
            {
                "name": self.current_model_name,
                "type": model_info.get("type", "unknown"),
                "description": model_info.get("description", ""),
                "classes": model_info.get("classes", 0),
                "slug": model_info.get("slug", ""),
                "active": self.pipeline_config.get("ai", False),
                "loading": self._model_loading,
                "error": self._model_load_error,
            }
        )
        self.ai_current_pub.publish(msg_curr)

        models_list = {}
        for name, info in AVAILABLE_MODELS.items():
            models_list[name] = {
                "type": info.get("type", "unknown"),
                "description": info.get("description", ""),
                "classes": info.get("classes", 0),
                "slug": info.get("slug", ""),
            }

        msg_avail = String()
        msg_avail.data = json.dumps(models_list)
        self.ai_available_pub.publish(msg_avail)

    def imu_config_callback(self, msg):
        """Handle IMU configuration: {"frequency": 100}."""
        try:
            config = json.loads(msg.data)
            if "frequency" in config:
                requested = int(config["frequency"])
                actual = self._validate_imu_frequency(requested)

                if actual != requested:
                    self.get_logger().warning(
                        f"IMU frequency {requested}Hz not supported, using {actual}Hz "
                        f"(BMI270 valid: {BMI270_VALID_FREQUENCIES})"
                    )

                if actual != self.imu_actual_freq:
                    self.imu_freq = requested
                    self.imu_actual_freq = actual
                    self._imu_unavailable = False
                    self.get_logger().info(f"IMU frequency set to {actual}Hz")

                    if self.pipeline_config.get("imu"):
                        self._force_rebuild = True
                        self.check_demand()
        except Exception as e:
            self.get_logger().error(f"Invalid IMU config: {e}")

    def ai_config_callback(self, msg):
        """
        Handle AI config:
        {
            "model": "name",
            "confidence": 0.5,
            "segmentation_mode": "bbox" | "mask",
            "segmentation_target_class": 15  # For mask mode
        }
        """
        try:
            data = (
                json.loads(msg.data)
                if msg.data.startswith("{")
                else {"model": msg.data.strip()}
            )
            rebuild_needed = False

            # Model change.
            if "model" in data:
                model_name = data["model"]
                if (
                    model_name in AVAILABLE_MODELS
                    and model_name != self.current_model_name
                ):
                    self.get_logger().info(f"AI Config: Switching to {model_name}")
                    self.current_model_name = model_name
                    self._ai_failed_model = None
                    self._model_load_error = None
                    rebuild_needed = True
                elif model_name not in AVAILABLE_MODELS:
                    self.get_logger().error(f"Unknown model: {model_name}")

            # Confidence change.
            if "confidence" in data:
                self.ai_confidence = float(data["confidence"])
                rebuild_needed = True

            # Segmentation mode change (no rebuild needed).
            if "segmentation_mode" in data:
                mode = data["segmentation_mode"]
                if mode in ["bbox", "mask"]:
                    self.segmentation_mode = mode
                    self.get_logger().info(f"Segmentation mode set to: {mode}")
                else:
                    self.get_logger().error(
                        f"Invalid segmentation_mode: {mode}. Use 'bbox' or 'mask'"
                    )

            # Segmentation target class.
            if "segmentation_target_class" in data:
                target = data["segmentation_target_class"]
                if target is None or isinstance(target, int):
                    self.segmentation_target_class = target
                    self.get_logger().info(
                        f"Segmentation target class set to: {target}"
                    )
                else:
                    self.get_logger().error(
                        f"Invalid segmentation_target_class: {target}. "
                        "Use integer or null"
                    )

            if rebuild_needed and self.pipeline_config.get("ai"):
                self._force_rebuild = True
                self.check_demand()

        except Exception as e:
            self.get_logger().error(f"Invalid AI config: {e}")

    def camera_config_callback(self, msg):
        """Handle camera/video config: {"quality": 80, "resolution": [1280, 720]}."""
        try:
            config = json.loads(msg.data)

            if "quality" in config:
                self.quality_factor = max(1, min(100, int(config["quality"])))

            if "resolution" in config:
                width, height = config["resolution"]
                if width != self.preview_width or height != self.preview_height:
                    # Frames are resized on the host, so no pipeline rebuild here.
                    self.preview_width = width
                    self.preview_height = height

            self.get_logger().info(
                f"Camera config updated: quality={self.quality_factor}, "
                f"resolution={self.preview_width}x{self.preview_height}"
            )

        except Exception as e:
            self.get_logger().error(f"Invalid camera config: {e}")

    def timer_period_callback(self, msg):
        self.timer_period = msg.data
        self.timer.cancel()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def quality_factor_callback(self, msg):
        self.quality_factor = msg.data

    def preview_size_callback(self, msg):
        self.preview_width, self.preview_height = msg.data

        if self.pipeline is not None:
            self.pipeline.stop()

        self.init_pipeline()

    def destroy_node(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        super().destroy_node()


def spin_camera(times):
    cnt = times
    if cnt == 0:
        print(
            "Couldn't restart camera due to displayed error/s, publishing error message"
        )
        rclpy.spin(error_publisher)
    else:
        camera_node = None
        try:
            camera_node = CameraNode()
            rclpy.spin(camera_node)
        except Exception as exc:
            error_publisher.timer_callback()
            print(exc)
        finally:
            if camera_node is not None:
                camera_node.destroy_node()
                print("camera_node destroyed")
            cnt = times - 1
            print("Retry starting camera..." + str(cnt))
            spin_camera(cnt)
    return


def main(args=None):
    rclpy.init()
    global error_publisher
    error_publisher = ErrorPublisher()
    print("Starting camera")
    spin_camera(3)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
