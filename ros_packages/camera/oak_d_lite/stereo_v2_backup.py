#!/usr/bin/python3
"""
OAK-D Lite Unified Node - DepthAI v3 Integration

Features:
- MJPEG Video Encoding - hardware accelerated (use CBOR subscription for binary transfer)
- AI Inference using Luxonis Model Hub - on-demand activation
- IMU Data (BMI270 6-axis) - accelerometer + gyroscope streaming
- Dynamic Model Selection - switch models at runtime
- RLE-encoded segmentation masks for efficient transmission

Uses DepthAI v3 API with NNModelDescription for model loading from Luxonis Model Hub.
All features are on-demand: only active when subscribed.
"""
import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import depthai as dai
import rclpy
import numpy as np

from datatypes.srv import GetCameraImage, SwitchModel
from rclpy.node import Node
from std_msgs.msg import String, Float64, Int32, Int32MultiArray
from sensor_msgs.msg import Imu, CompressedImage
from geometry_msgs.msg import Vector3Stamped

# ============== MODEL REGISTRY (DepthAI v3 Model Hub) ==============
# Format: "luxonis/model-name:variant" for Model Hub
# All models are optimized for RVC2 (OAK-D Lite)

AVAILABLE_MODELS = {
    # ============== OBJECT DETECTION ==============
    "yolov8n": {
        "type": "detection",
        "slug": "luxonis/yolov8-nano:coco-512x288",
        "description": "YOLOv8 Nano - fast & accurate object detection",
        "classes": 80,  # COCO classes
        "node_type": "DetectionNetwork",
    },
    "yolov6n": {
        "type": "detection",
        "slug": "luxonis/yolov6-nano:r2-coco-512x288",
        "description": "YOLOv6 Nano - efficient detection",
        "classes": 80,
        "node_type": "DetectionNetwork",
    },
    "yolov10n": {
        "type": "detection",
        "slug": "luxonis/yolov10-nano:coco-512x288",
        "description": "YOLOv10 Nano - latest YOLO architecture",
        "classes": 80,
        "node_type": "DetectionNetwork",
    },
    
    # ============== FACE DETECTION ==============
    "face": {
        "type": "detection",
        "slug": "luxonis/scrfd:2.5g-kps-640x640",
        "description": "SCRFD face detection with keypoints",
        "classes": 1,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "ImgDetectionsExtended",  # Has keypoints
    },
    
    # ============== PERSON DETECTION ==============
    "person": {
        "type": "detection",
        "slug": "luxonis/scrfd-person-detection:25g-640x640",
        "description": "SCRFD person detection",
        "classes": 1,
        "node_type": "DetectionNetwork",
    },
    
    # ============== POSE ESTIMATION ==============
    "pose_yolo": {
        "type": "pose",
        "slug": "luxonis/yolov8-nano-pose-estimation:coco-512x288",
        "description": "YOLOv8 Nano pose estimation (17 keypoints)",
        "keypoints": 17,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Keypoints",
    },
    "pose_hrnet": {
        "type": "pose",
        "slug": "luxonis/lite-hrnet:18-coco-288x384",
        "description": "Lite-HRNet - accurate pose estimation",
        "keypoints": 17,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Keypoints",
    },
    
    # ============== HAND TRACKING ==============
    "hand": {
        "type": "hand",
        "slug": "luxonis/mediapipe-hand-landmarker:224x224",
        "description": "MediaPipe hand landmark detection (21 keypoints)",
        "keypoints": 21,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Keypoints",
    },
    
    # ============== INSTANCE SEGMENTATION ==============
    "segmentation": {
        "type": "instance-segmentation",
        "slug": "luxonis/yolov8-instance-segmentation-nano:coco-512x288",
        "description": "YOLOv8 Nano instance segmentation",
        "classes": 80,
        "node_type": "ParsingNeuralNetwork",
        "output_type": "ImgDetectionsExtended",  # Detection + masks
    },
    
    # ============== GAZE ESTIMATION ==============
    "gaze": {
        "type": "gaze",
        "slug": "luxonis/l2cs-net:448x448",
        "description": "L2CS-Net gaze estimation",
        "node_type": "ParsingNeuralNetwork",
        "output_type": "Predictions",  # Yaw/pitch predictions
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

# Default model to use
DEFAULT_MODEL = "yolov8n"

# BMI270 valid frequencies (sensor rounds DOWN to nearest)
BMI270_VALID_FREQUENCIES = [25, 50, 100, 200, 250]


def rle_encode(binary_mask: np.ndarray) -> dict:
    """
    Run-length encode a binary mask for efficient transmission.

    Args:
        binary_mask: 2D numpy array with 0s and 1s (or 0s and 255s)

    Returns:
        dict with 'size' (width, height) and 'counts' (run lengths starting with 0-count)

    Format: counts alternate between 0-pixels and 1-pixels, starting with 0-count.
    Example: [3, 5, 2, 10] means 3 zeros, 5 ones, 2 zeros, 10 ones

    This is compatible with COCO RLE format and pycocotools.
    """
    h, w = binary_mask.shape
    # Flatten in column-major (Fortran) order for COCO compatibility
    pixels = (binary_mask.flatten(order='F') > 0).astype(np.uint8)

    # Find transitions
    pixels_padded = np.concatenate([[0], pixels, [0]])
    transitions = np.where(pixels_padded[1:] != pixels_padded[:-1])[0]

    # Calculate run lengths
    counts = np.diff(transitions).tolist()

    # If mask starts with 1, prepend a 0-count
    if len(counts) > 0 and pixels[0] == 1:
        counts = [0] + counts

    return {
        "size": [h, w],  # height, width (COCO format)
        "counts": counts
    }


def rle_decode(rle: dict) -> np.ndarray:
    """
    Decode RLE back to binary mask (for verification/testing).

    Args:
        rle: dict with 'size' [h, w] and 'counts' (run lengths)

    Returns:
        2D numpy array binary mask
    """
    h, w = rle["size"]
    counts = rle["counts"]

    pixels = []
    val = 0  # Start with 0s
    for count in counts:
        pixels.extend([val] * count)
        val = 1 - val  # Toggle between 0 and 1

    # Reshape in column-major order
    mask = np.array(pixels, dtype=np.uint8).reshape((h, w), order='F')
    return mask


class ErrorPublisher(Node):
    def __init__(self):
        super().__init__("error_publisher")
        self.publisher_ = self.create_publisher(String, "camera_topic", 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = "Camera not available"
        self.publisher_.publish(msg)


class OakUnifiedNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        
        # Check DepthAI version
        self._check_depthai_version()
        
        # ============== PUBLISHERS ==============
        # Video - MJPEG frames (subscribe with CBOR compression for binary transfer)
        # Legacy: camera_topic (base64 JPEG string) - kept for backward compatibility
        self.rgb_pub = self.create_publisher(String, "camera_topic", 10)
        # New: camera/image (binary JPEG via CompressedImage, use CBOR subscription)
        self.camera_image_pub = self.create_publisher(CompressedImage, "camera/image", 10)

        # IMU (follows existing imu/* namespace)
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.imu_accel_pub = self.create_publisher(Vector3Stamped, "imu/accelerometer", 10)
        self.imu_gyro_pub = self.create_publisher(Vector3Stamped, "imu/gyroscope", 10)

        # AI (follows existing ai/* namespace pattern)
        self.ai_pub = self.create_publisher(String, "ai/detections", 10)
        self.ai_available_pub = self.create_publisher(String, "ai/available_models", 10)
        self.ai_current_pub = self.create_publisher(String, "ai/current_model", 10)
        self.ai_status_pub = self.create_publisher(String, "ai/status", 10)  # For model loading status

        # ============== SUBSCRIBERS ==============
        self.create_subscription(Float64, "timer_period_topic", self.timer_period_callback, 10)
        self.create_subscription(Int32, "quality_factor_topic", self.quality_factor_callback, 10)
        self.create_subscription(Int32MultiArray, "size_topic", self.preview_size_callback, 10)
        self.create_subscription(String, "imu/config", self.imu_config_callback, 10)
        self.create_subscription(String, "ai/config", self.ai_config_callback, 10)
        self.create_subscription(String, "camera/config", self.camera_config_callback, 10)

        # ============== SERVICES ==============
        self.create_service(GetCameraImage, "get_camera_image", self.get_camera_image_callback)
        self.create_service(SwitchModel, "camera_node/switch_model", self.switch_model_callback)

        # ============== VIDEO CONFIG ==============
        self.preview_width = 1280
        self.preview_height = 720
        self.quality_factor = 80
        self.video_fps = 30
        self.video_bitrate = 0  # 0 = auto (use quality_factor for MJPEG)
        self.current_image = ""
        
        # ============== IMU CONFIG ==============
        self.imu_freq = 100
        self.imu_actual_freq = 100  # After BMI270 rounding
        
        # ============== AI CONFIG ==============
        self.current_model_name = DEFAULT_MODEL
        self.ai_confidence = 0.5
        
        # Segmentation mode: "bbox" (bounding boxes for all classes) or "mask" (binary mask for target_class)
        self.segmentation_mode = "bbox"
        self.segmentation_target_class = None  # If set, return binary mask for this class
        
        # ============== DEVICE STATE ==============
        self.device: Optional[dai.Device] = None
        self.queues: Dict[str, Any] = {}
        self.current_pipeline_config = {
            'video': False,  # MJPEG camera frames
            'ai': False,
            'imu': False
        }
        
        # Frame counter for AI output
        self._frame_count = 0
        self._ai_inference_start = 0.0
        
        # ============== TIMERS ==============
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.status_timer = self.create_timer(1.0, self.status_timer_callback)
        
        # Flag to force pipeline rebuild (e.g., after model change)
        self._force_rebuild = False
        
        # Model loading state for status feedback
        self._model_loading = False
        self._model_load_error: Optional[str] = None
        
        self.get_logger().info("OakUnifiedNode initialized (DepthAI v3 + MJPEG + AI + IMU). Waiting for subscribers...")
    
    def _publish_status(self, state: str, message: str = "", model: str = ""):
        """Publish AI status update to ai/status topic"""
        if not model:
            model = self.current_model_name
        status = {
            "state": state,  # "idle", "loading", "ready", "error"
            "model": model,
            "message": message,
            "timestamp": time.time()
        }
        msg = String()
        msg.data = json.dumps(status)
        self.ai_status_pub.publish(msg)

    def _check_depthai_version(self):
        """Verify we have DepthAI v3+"""
        version = getattr(dai, '__version__', '0.0.0')
        major = int(version.split('.')[0])
        if major < 3:
            self.get_logger().warning(
                f"DepthAI v{version} detected. This node is optimized for DepthAI v3.0.0+. "
                "Some features may not work correctly."
            )
        else:
            self.get_logger().info(f"DepthAI version: {version}")

    def check_demand(self):
        """Check all subscriber counts and determine need"""
        # Video needed if legacy (base64) or new (CBOR binary) subscribers exist
        need_video = (self.rgb_pub.get_subscription_count() > 0 or
                      self.camera_image_pub.get_subscription_count() > 0)
        need_ai = self.ai_pub.get_subscription_count() > 0
        need_imu = (self.imu_pub.get_subscription_count() > 0 or
                    self.imu_accel_pub.get_subscription_count() > 0 or
                    self.imu_gyro_pub.get_subscription_count() > 0)

        new_config = {
            'video': need_video,
            'ai': need_ai,
            'imu': need_imu
        }
        
        # Determine if rebuild is needed
        features_changed = False

        # If we have no device, and we need something, rebuild
        if self.device is None and any(new_config.values()):
            features_changed = True

        # If we have device, check if feature set changed
        elif self.device is not None:
            if new_config != self.current_pipeline_config:
                features_changed = True

            # Also close device if all demands drop to zero (save power/heat)
            if not any(new_config.values()):
                features_changed = True

        # Also rebuild if forced (e.g., model change, config change)
        if self._force_rebuild:
            features_changed = True
            self._force_rebuild = False
        
        if features_changed:
            self.get_logger().info(f"Demand changed: {new_config}. Rebuilding pipeline...")
            self._rebuild_pipeline(new_config)

    def _get_model_description(self, model_name: str) -> dai.NNModelDescription:
        """Get NNModelDescription for a model from the registry"""
        model_info = AVAILABLE_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        slug = model_info.get('slug')
        if not slug:
            raise ValueError(f"Model {model_name} has no slug defined")
        
        self.get_logger().info(f"Creating model description for: {slug}")
        return dai.NNModelDescription(slug)

    def _rebuild_pipeline(self, config):
        """Rebuild the pipeline with the current configuration"""
        if self.device:
            self.device.close()
            self.device = None
            self.queues = {}

        if not any(config.values()):
            self.current_pipeline_config = config
            self._publish_status("idle", "No features active")
            return

        # Track if AI was successfully configured
        ai_configured = False
        self._model_loading = config['ai']
        self._model_load_error = None

        if config['ai']:
            self._publish_status("loading", f"Loading model {self.current_model_name}...")

        try:
            # Create pipeline (don't use context manager - we need it to persist!)
            pipeline = dai.Pipeline()
            
            # --- Camera (MJPEG) & AI ---
            needs_camera = config['video'] or config['ai']

            if needs_camera:
                camRgb = pipeline.create(dai.node.ColorCamera)
                camRgb.setPreviewSize(self.preview_width, self.preview_height)
                camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                camRgb.setInterleaved(False)
                camRgb.setFps(self.video_fps)

                # Video Output - MJPEG encoder for individual frames
                if config['video']:
                    # Use hardware MJPEG encoder - each frame is independent JPEG
                    mjpeg_enc = pipeline.create(dai.node.VideoEncoder)
                    mjpeg_enc.setDefaultProfilePreset(self.video_fps, dai.VideoEncoderProperties.Profile.MJPEG)
                    mjpeg_enc.setQuality(self.quality_factor)
                    camRgb.video.link(mjpeg_enc.input)

                    xoutVideo = pipeline.create(dai.node.XLinkOut)
                    xoutVideo.setStreamName("video")
                    mjpeg_enc.bitstream.link(xoutVideo.input)

                # AI Output - Use DepthAI v3 Model Hub
                if config['ai']:
                    try:
                        model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
                        node_type = model_info.get('node_type', 'DetectionNetwork')

                        self.get_logger().info(
                            f"Loading model: {self.current_model_name} "
                            f"(type={model_info.get('type')}, node={node_type}) - "
                            "this may take a moment if downloading..."
                        )
                        
                        # Get model description - this may trigger download from Model Hub
                        model_desc = self._get_model_description(self.current_model_name)

                        if node_type == 'DetectionNetwork':
                            # Simple detection - auto-parsed to dai.ImgDetections
                            nn = pipeline.create(dai.node.DetectionNetwork).build(
                                camRgb.preview, model_desc
                            )
                            nn.input.setBlocking(False)
                            
                            xoutNn = pipeline.create(dai.node.XLinkOut)
                            xoutNn.setStreamName("nn")
                            nn.out.link(xoutNn.input)
                            ai_configured = True
                            
                        elif node_type == 'ParsingNeuralNetwork':
                            # Complex models with custom parsers (depthai-nodes)
                            try:
                                from depthai_nodes import ParsingNeuralNetwork
                                nn = ParsingNeuralNetwork.build(camRgb.preview, model_desc)
                                nn.input.setBlocking(False)
                                
                                xoutNn = pipeline.create(dai.node.XLinkOut)
                                xoutNn.setStreamName("nn")
                                nn.out.link(xoutNn.input)
                                ai_configured = True
                            except ImportError:
                                self.get_logger().error(
                                    "depthai-nodes not installed. Install with: "
                                    "pip install depthai-nodes"
                                )
                                # Fall back to NeuralNetwork for raw output
                                nn = pipeline.create(dai.node.NeuralNetwork)
                                nn.setNNModelDescription(model_desc)
                                nn.input.setBlocking(False)
                                camRgb.preview.link(nn.input)
                                
                                xoutNn = pipeline.create(dai.node.XLinkOut)
                                xoutNn.setStreamName("nn")
                                nn.out.link(xoutNn.input)
                                ai_configured = True
                        else:
                            # Generic NeuralNetwork fallback
                            nn = pipeline.create(dai.node.NeuralNetwork)
                            nn.setNNModelDescription(model_desc)
                            nn.input.setBlocking(False)
                            camRgb.preview.link(nn.input)
                            
                            xoutNn = pipeline.create(dai.node.XLinkOut)
                            xoutNn.setStreamName("nn")
                            nn.out.link(xoutNn.input)
                            ai_configured = True

                    except Exception as e:
                        error_msg = f"Failed to load model {self.current_model_name}: {e}"
                        self.get_logger().error(error_msg)
                        self._model_load_error = str(e)
                        self._publish_status("error", error_msg)
                        # AI failed, but we can still run video/IMU
                        ai_configured = False

            # --- IMU ---
            if config['imu']:
                imu = pipeline.create(dai.node.IMU)
                sensors = [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW]
                imu.enableIMUSensor(sensors, self.imu_actual_freq)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)

                xoutImu = pipeline.create(dai.node.XLinkOut)
                xoutImu.setStreamName("imu")
                imu.out.link(xoutImu.input)

            # Start the pipeline with the device
            self.device = dai.Device(pipeline)
            
            # Update config to reflect what was actually configured
            actual_config = config.copy()
            if config['ai'] and not ai_configured:
                actual_config['ai'] = False
                self.get_logger().warning("AI was requested but failed to configure")
            
            self.current_pipeline_config = actual_config
            self._model_loading = False

            # Setup queues
            if actual_config['video']:
                self.queues['video'] = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
            if actual_config['ai']:
                self.queues['nn'] = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
                # Publish success status
                self._publish_status("ready", f"Model {self.current_model_name} loaded successfully")
            if actual_config['imu']:
                self.queues['imu'] = self.device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            active_features = [k for k, v in actual_config.items() if v]
            self.get_logger().info(f"Pipeline started: {active_features}")
            
        except Exception as e:
            error_msg = f"Failed to start pipeline: {e}"
            self.get_logger().error(error_msg)
            self._model_loading = False
            self._model_load_error = str(e)
            self._publish_status("error", error_msg)
            self.device = None
            self.current_pipeline_config = {'video': False, 'ai': False, 'imu': False}

    def timer_callback(self):
        # 1. Check demand and rebuild if necessary
        try:
            self.check_demand()
        except Exception as e:
            self.get_logger().error(f"Error in check_demand: {e}")

        if not self.device:
            return

        # 2. Process MJPEG video frames
        if 'video' in self.queues:
            q = self.queues['video']
            while q.has():
                packet = q.get()
                jpeg_data = packet.getData().tobytes()

                # Publish to new binary topic (use CBOR subscription for efficiency)
                img_msg = CompressedImage()
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "oak_camera"
                img_msg.format = "jpeg"
                img_msg.data = jpeg_data
                self.camera_image_pub.publish(img_msg)

                # Also publish to legacy base64 topic for backward compatibility
                if self.rgb_pub.get_subscription_count() > 0:
                    legacy_msg = String()
                    legacy_msg.data = base64.b64encode(jpeg_data).decode('utf-8')
                    self.current_image = legacy_msg.data
                    self.rgb_pub.publish(legacy_msg)

        # 3. Process AI
        if 'nn' in self.queues:
            q = self.queues['nn']
            while q.has():
                self._frame_count += 1

                in_data = q.get()
                model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
                model_type = model_info.get('type', 'detection')
                output_type = model_info.get('output_type', 'ImgDetections')

                # Calculate latency
                try:
                    frame_ts = in_data.getTimestamp()
                    now = dai.Clock.now()
                    latency_ms = (now - frame_ts).total_seconds() * 1000
                except Exception:
                    latency_ms = 0.0

                result = self._format_ai_result(in_data, model_type, output_type, model_info)

                # Build enhanced output
                output = {
                    "model": self.current_model_name,
                    "type": model_type,
                    "frame_id": self._frame_count,
                    "timestamp_ns": self.get_clock().now().nanoseconds,
                    "latency_ms": round(latency_ms, 2),
                    "result": result
                }
                
                msg = String()
                msg.data = json.dumps(output)
                self.ai_pub.publish(msg)
                
        # 4. Process IMU
        if 'imu' in self.queues:
            q = self.queues['imu']
            while q.has():
                imu_packets = q.get().packets
                for packet in imu_packets:
                    # Publish Imu msg
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
                    
                    # Publish components
                    accel_msg = Vector3Stamped()
                    accel_msg.header = imu_msg.header
                    accel_msg.vector = imu_msg.linear_acceleration
                    self.imu_accel_pub.publish(accel_msg)
                    
                    gyro_msg = Vector3Stamped()
                    gyro_msg.header = imu_msg.header
                    gyro_msg.vector = imu_msg.angular_velocity
                    self.imu_gyro_pub.publish(gyro_msg)

    def _format_ai_result(self, in_data, model_type: str, output_type: str, model_info: dict) -> dict:
        """Format AI inference result based on model type and output type"""
        
        # Handle DepthAI v3 parsed outputs
        if output_type == 'ImgDetections' or isinstance(in_data, dai.ImgDetections):
            # Standard detections
            return self._format_detections(in_data)
        
        # Try depthai-nodes output types
        try:
            from depthai_nodes.ml.messages import (
                ImgDetectionsExtended,
                Keypoints,
                Lines,
                Predictions
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
            pass  # depthai-nodes not installed, fall through to raw handling
        
        # Fallback: Handle based on model_type for raw NeuralNetwork output
        if model_type == 'detection':
            return self._format_detections(in_data)
        elif model_type == 'pose':
            return self._format_pose_raw(in_data, model_info)
        elif model_type == 'instance-segmentation':
            return self._format_instance_seg_raw(in_data, model_info)
        elif model_type == 'gaze':
            return self._format_gaze_raw(in_data)
        elif model_type == 'lines':
            return self._format_lines_raw(in_data)
        elif model_type == 'hand':
            return self._format_hand_raw(in_data, model_info)
        else:
            return {"raw": "unsupported model type", "type": model_type}
    
    def _format_detections(self, in_data) -> dict:
        """Format standard ImgDetections"""
        detections = []
        try:
            for det in in_data.detections:
                detections.append({
                    "label": det.label,
                    "confidence": round(det.confidence, 4),
                    "bbox": {
                        "xmin": round(det.xmin, 4),
                        "ymin": round(det.ymin, 4),
                        "xmax": round(det.xmax, 4),
                        "ymax": round(det.ymax, 4)
                    }
                })
        except Exception as e:
            return {"error": str(e)}
        
        return {"detections": detections, "count": len(detections)}
    
    def _format_detections_extended(self, in_data, model_info: dict) -> dict:
        """Format ImgDetectionsExtended (detections with keypoints/masks)"""
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
                        "ymax": round(det.ymax, 4)
                    }
                }
                
                # Add keypoints if available (e.g., face landmarks)
                if hasattr(det, 'keypoints') and det.keypoints:
                    kps = []
                    for i, kp in enumerate(det.keypoints):
                        kps.append({
                            "id": i,
                            "x": round(kp.x, 4),
                            "y": round(kp.y, 4),
                            "confidence": round(getattr(kp, 'confidence', 1.0), 4)
                        })
                    det_dict["keypoints"] = kps
                
                # Add mask if available (instance segmentation)
                if hasattr(det, 'mask') and det.mask is not None:
                    if self.segmentation_mode == "mask":
                        # RLE encode the mask
                        mask_array = np.array(det.mask)
                        if mask_array.ndim == 2:
                            det_dict["mask_rle"] = rle_encode(mask_array)
                    else:
                        # Just note that mask is available
                        det_dict["has_mask"] = True
                
                detections.append(det_dict)
                
        except Exception as e:
            return {"error": str(e)}
        
        return {"detections": detections, "count": len(detections)}
    
    def _format_keypoints(self, in_data, model_info: dict) -> dict:
        """Format Keypoints output (pose/hand estimation)"""
        try:
            keypoints = []
            num_keypoints = model_info.get('keypoints', 17)
            
            for i, kp in enumerate(in_data.keypoints[:num_keypoints]):
                keypoints.append({
                    "id": i,
                    "x": round(kp.x, 4),
                    "y": round(kp.y, 4),
                    "confidence": round(getattr(kp, 'confidence', 1.0), 4)
                })
            
            detected_count = len([k for k in keypoints if k['confidence'] > 0.3])
            
            return {
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "detected_count": detected_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_lines(self, in_data) -> dict:
        """Format Lines output (line segment detection)"""
        try:
            lines = []
            for line in in_data.lines:
                lines.append({
                    "start": {"x": round(line.start.x, 4), "y": round(line.start.y, 4)},
                    "end": {"x": round(line.end.x, 4), "y": round(line.end.y, 4)},
                    "confidence": round(getattr(line, 'confidence', 1.0), 4)
                })
            
            return {"lines": lines, "count": len(lines)}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_predictions(self, in_data, model_type: str) -> dict:
        """Format Predictions output (gaze, classification, etc.)"""
        try:
            if model_type == 'gaze':
                # L2CS-Net outputs yaw and pitch
                predictions = in_data.predictions
                if len(predictions) >= 2:
                    return {
                        "yaw": round(float(predictions[0]), 4),
                        "pitch": round(float(predictions[1]), 4),
                        "unit": "radians"
                    }
            
            # Generic predictions
            return {
                "predictions": [round(float(p), 4) for p in in_data.predictions]
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ============== RAW OUTPUT HANDLERS (fallback when depthai-nodes not available) ==============
    
    def _format_pose_raw(self, in_data, model_info: dict) -> dict:
        """Format raw pose estimation output"""
        try:
            layer_data = in_data.getFirstLayerFp16()
            num_keypoints = model_info.get('keypoints', 17)
            
            keypoints = []
            if hasattr(layer_data, 'shape'):
                heatmaps = np.array(layer_data)
                
                if len(heatmaps.shape) == 3:
                    for i in range(min(heatmaps.shape[0], num_keypoints)):
                        heatmap = heatmaps[i]
                        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        confidence = float(heatmap[max_idx])
                        y_norm = max_idx[0] / heatmap.shape[0]
                        x_norm = max_idx[1] / heatmap.shape[1]
                        
                        keypoints.append({
                            "id": i,
                            "x": round(x_norm, 4),
                            "y": round(y_norm, 4),
                            "confidence": round(confidence, 4)
                        })
            
            return {
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "detected_count": len([k for k in keypoints if k['confidence'] > 0.3]),
                "note": "raw_output"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_instance_seg_raw(self, in_data, model_info: dict) -> dict:
        """Format raw instance segmentation output"""
        try:
            # Try to get detections if available
            if hasattr(in_data, 'detections'):
                return self._format_detections(in_data)
            
            return {
                "note": "Install depthai-nodes for full instance segmentation support",
                "raw_layers": in_data.getLayerNames() if hasattr(in_data, 'getLayerNames') else []
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_gaze_raw(self, in_data) -> dict:
        """Format raw gaze estimation output"""
        try:
            data = in_data.getFirstLayerFp16()
            if len(data) >= 2:
                return {
                    "yaw": round(float(data[0]), 4),
                    "pitch": round(float(data[1]), 4),
                    "unit": "radians",
                    "note": "raw_output"
                }
            return {"raw": list(data[:10])}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_lines_raw(self, in_data) -> dict:
        """Format raw line detection output"""
        try:
            return {
                "note": "Install depthai-nodes for line detection parsing",
                "raw_layers": in_data.getLayerNames() if hasattr(in_data, 'getLayerNames') else []
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _format_hand_raw(self, in_data, model_info: dict) -> dict:
        """Format raw hand landmark output"""
        try:
            layer_data = in_data.getFirstLayerFp16()
            num_keypoints = model_info.get('keypoints', 21)
            
            # MediaPipe hand outputs [x, y, z] * 21 keypoints = 63 values
            keypoints = []
            data = np.array(layer_data)
            
            for i in range(min(len(data) // 3, num_keypoints)):
                keypoints.append({
                    "id": i,
                    "x": round(float(data[i * 3]), 4),
                    "y": round(float(data[i * 3 + 1]), 4),
                    "z": round(float(data[i * 3 + 2]), 4),
                })
            
            return {
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "note": "raw_output"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _validate_imu_frequency(self, requested_freq: int) -> int:
        """Validate and round IMU frequency to BMI270 supported values"""
        valid = [f for f in BMI270_VALID_FREQUENCIES if f <= requested_freq]
        if not valid:
            return BMI270_VALID_FREQUENCIES[0]
        return max(valid)

    # ============== CALLBACKS ==============
    
    def get_camera_image_callback(self, request, response):
        response.image_base64 = self.current_image
        return response

    def switch_model_callback(self, request, response):
        model_name = request.model_name
        
        if model_name not in AVAILABLE_MODELS:
            response.success = False
            response.message = f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
            return response
        
        if model_name == self.current_model_name:
            response.success = True
            response.message = f"Already using model: {model_name}"
            return response
        
        self.get_logger().info(f"Switching model to: {model_name}")
        self.current_model_name = model_name
        
        # Trigger rebuild if AI is active
        if self.current_pipeline_config.get('ai'):
            self._force_rebuild = True
            # Publish status before rebuild (service call returns immediately)
            self._publish_status("loading", f"Switching to model {model_name}...")
            self.check_demand()
            
            # Check if rebuild succeeded (it happens synchronously in check_demand)
            if self._model_load_error:
                response.success = False
                response.message = f"Model switch failed: {self._model_load_error}"
                return response
             
        response.success = True
        response.message = f"Switched to {model_name}. Subscribe to ai/status for loading progress."
        return response
        
    def imu_config_callback(self, msg):
        """Handle IMU configuration: {"frequency": 100}"""
        try:
            config = json.loads(msg.data)
            if 'frequency' in config:
                requested = int(config['frequency'])
                actual = self._validate_imu_frequency(requested)
                
                if actual != requested:
                    self.get_logger().warning(
                        f"IMU frequency {requested}Hz not supported, using {actual}Hz "
                        f"(BMI270 valid: {BMI270_VALID_FREQUENCIES})"
                    )
                
                if actual != self.imu_actual_freq:
                    self.imu_freq = requested
                    self.imu_actual_freq = actual
                    self.get_logger().info(f"IMU frequency set to {actual}Hz")
                    
                    if self.current_pipeline_config.get('imu'):
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
            data = json.loads(msg.data) if msg.data.startswith('{') else {"model": msg.data.strip()}
            rebuild_needed = False
            
            # Model change
            if 'model' in data:
                model_name = data['model']
                if model_name in AVAILABLE_MODELS and model_name != self.current_model_name:
                    self.get_logger().info(f"AI Config: Switching to {model_name}")
                    self.current_model_name = model_name
                    rebuild_needed = True
                elif model_name not in AVAILABLE_MODELS:
                    self.get_logger().error(f"Unknown model: {model_name}")
            
            # Confidence change
            if 'confidence' in data:
                self.ai_confidence = float(data['confidence'])
                rebuild_needed = True
            
            # Segmentation mode change (no rebuild needed)
            if 'segmentation_mode' in data:
                mode = data['segmentation_mode']
                if mode in ['bbox', 'mask']:
                    self.segmentation_mode = mode
                    self.get_logger().info(f"Segmentation mode set to: {mode}")
                else:
                    self.get_logger().error(f"Invalid segmentation_mode: {mode}. Use 'bbox' or 'mask'")
            
            # Segmentation target class
            if 'segmentation_target_class' in data:
                target = data['segmentation_target_class']
                if target is None or isinstance(target, int):
                    self.segmentation_target_class = target
                    self.get_logger().info(f"Segmentation target class set to: {target}")
                else:
                    self.get_logger().error(f"Invalid segmentation_target_class: {target}. Use integer or null")
                
            if rebuild_needed and self.current_pipeline_config.get('ai'):
                self._force_rebuild = True
                self.check_demand()
                
        except Exception as e:
            self.get_logger().error(f"Invalid AI config: {e}")
    
    def camera_config_callback(self, msg):
        """Handle camera/video config: {"fps": 30, "quality": 80, "resolution": [1280, 720]}"""
        try:
            config = json.loads(msg.data)
            rebuild_needed = False
            
            if 'fps' in config:
                new_fps = int(config['fps'])
                if new_fps != self.video_fps:
                    self.video_fps = new_fps
                    rebuild_needed = True
                    
            if 'quality' in config:
                new_quality = int(config['quality'])
                if new_quality != self.quality_factor:
                    self.quality_factor = max(1, min(100, new_quality))
                    rebuild_needed = True
                    
            if 'resolution' in config:
                w, h = config['resolution']
                if w != self.preview_width or h != self.preview_height:
                    self.preview_width = w
                    self.preview_height = h
                    rebuild_needed = True
            
            if rebuild_needed:
                self.get_logger().info(
                    f"Camera config updated: fps={self.video_fps}, "
                    f"resolution={self.preview_width}x{self.preview_height}"
                )
                if self.current_pipeline_config.get('video'):
                    self._force_rebuild = True
                    self.check_demand()
                    
        except Exception as e:
            self.get_logger().error(f"Invalid camera config: {e}")

    def status_timer_callback(self):
        # Publish current model with metadata
        model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
        msg_curr = String()
        msg_curr.data = json.dumps({
            "name": self.current_model_name,
            "type": model_info.get('type', 'unknown'),
            "description": model_info.get('description', ''),
            "classes": model_info.get('classes', 0),
            "slug": model_info.get('slug', ''),
            "active": self.current_pipeline_config.get('ai', False),
            "loading": self._model_loading,
            "error": self._model_load_error
        })
        self.ai_current_pub.publish(msg_curr)
        
        # Publish available models with metadata
        models_list = {}
        for name, info in AVAILABLE_MODELS.items():
            models_list[name] = {
                "type": info.get('type', 'unknown'),
                "description": info.get('description', ''),
                "classes": info.get('classes', 0),
                "slug": info.get('slug', '')
            }
        
        msg_avail = String()
        msg_avail.data = json.dumps(models_list)
        self.ai_available_pub.publish(msg_avail)

    def timer_period_callback(self, msg):
        self.timer_period = msg.data
        self.timer.cancel()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def quality_factor_callback(self, msg):
        self.quality_factor = msg.data

    def preview_size_callback(self, msg):
        self.preview_width, self.preview_height = msg.data
        if self.current_pipeline_config.get('video') or self.current_pipeline_config.get('ai'):
            self._force_rebuild = True
            self.check_demand()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OakUnifiedNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution failed: {e}")
        error_node = ErrorPublisher()
        rclpy.spin(error_node)
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
