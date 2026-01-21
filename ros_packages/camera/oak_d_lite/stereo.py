#!/usr/bin/python3
"""
OAK-D Lite Unified Node - DepthAI v3 Integration

Features:
- MJPEG Video Encoding - hardware accelerated (use CBOR subscription for binary transfer)
- AI Inference using Luxonis Model Hub - on-demand activation
- IMU Data (BMI270 6-axis) - accelerometer + gyroscope streaming
- Dynamic Model Selection - switch models at runtime
- RLE-encoded segmentation masks for efficient transmission

Uses DepthAI v3 API with:
- dai.node.Camera with .build() and .requestOutput()
- createOutputQueue() instead of XLinkOut nodes
- pipeline.start()/stop() instead of dai.Device(pipeline)
- DetectionNetwork.build(cameraNode, modelDescription)

All features are on-demand: only active when subscribed.
"""
import base64
import json
import time
import threading
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
# Slugs must match scripts/download_oak_models.py for pre-caching

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

# Valid IMU frequencies for BMI270 sensor
BMI270_VALID_FREQUENCIES = [25, 50, 100, 200, 400]


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
    
    return {
        "runs": runs,
        "values": values,
        "shape": list(mask.shape)
    }


class ErrorPublisher(Node):
    """Fallback node to publish errors when camera fails"""
    
    def __init__(self):
        super().__init__("camera_error")
        self.error_pub = self.create_publisher(String, "camera/error", 10)
        self.timer = self.create_timer(1.0, self._publish_error)
        
    def _publish_error(self):
        msg = String()
        msg.data = json.dumps({
            "error": "OAK-D Lite camera failed to initialize",
            "timestamp": time.time()
        })
        self.error_pub.publish(msg)


class OakUnifiedNode(Node):
    """
    Unified OAK-D Lite Node with DepthAI v3 API
    
    All features are on-demand - only active when there are subscribers.
    Uses v3 API patterns:
    - Pipeline context manager NOT used (need persistent pipeline)
    - Camera.build() + requestOutput()
    - createOutputQueue() instead of XLinkOut
    - VideoEncoder.build(output, frameRate, profile)
    - DetectionNetwork.build(cameraNode, modelDescription)
    """
    
    def __init__(self):
        super().__init__("oak_unified_node")
        
        # Check DepthAI version
        self._check_depthai_version()
        
        # Pipeline state
        self.pipeline: Optional[dai.Pipeline] = None
        self.queues: Dict[str, Any] = {}
        self.current_pipeline_config = {'video': False, 'ai': False, 'imu': False}
        self._force_rebuild = False
        self._pipeline_lock = threading.Lock()
        
        # Model state
        self.current_model_name = "yolov6n"
        self._model_loading = False
        self._model_load_error: Optional[str] = None
        
        # Camera settings
        self.preview_width = 640
        self.preview_height = 480
        self.video_fps = 30
        self.quality_factor = 80
        self.current_image = None  # For service
        
        # AI settings
        self.ai_confidence = 0.5
        self.segmentation_mode = "bbox"  # "bbox" or "mask"
        self.segmentation_target_class = None  # None = all classes
        self._frame_count = 0
        
        # IMU settings
        self.imu_freq = 100
        self.imu_actual_freq = self._validate_imu_frequency(100)
        
        # Timer
        self.timer_period = 0.03  # ~33Hz
        
        # ============== PUBLISHERS ==============
        # Video - binary (efficient)
        self.camera_image_pub = self.create_publisher(
            CompressedImage, "camera/image/compressed", 10
        )
        # Video - legacy base64 (backward compatibility)
        self.rgb_pub = self.create_publisher(String, "camera/rgb/image", 10)
        
        # AI
        self.ai_pub = self.create_publisher(String, "camera/ai/detections", 10)
        self.ai_current_pub = self.create_publisher(String, "camera/ai/current_model", 10)
        self.ai_available_pub = self.create_publisher(String, "camera/ai/available_models", 10)
        self.ai_status_pub = self.create_publisher(String, "camera/ai/status", 10)
        
        # IMU
        self.imu_pub = self.create_publisher(Imu, "camera/imu", 50)
        self.imu_accel_pub = self.create_publisher(
            Vector3Stamped, "camera/imu/accelerometer", 50
        )
        self.imu_gyro_pub = self.create_publisher(
            Vector3Stamped, "camera/imu/gyroscope", 50
        )
        
        # Error
        self.error_pub = self.create_publisher(String, "camera/error", 10)
        
        # ============== SUBSCRIBERS ==============
        self.create_subscription(
            String, "camera/ai/config", self.ai_config_callback, 10
        )
        self.create_subscription(
            String, "camera/video/config", self.camera_config_callback, 10
        )
        self.create_subscription(
            String, "camera/imu/config", self.imu_config_callback, 10
        )
        self.create_subscription(
            Float64, "camera/timer_period", self.timer_period_callback, 10
        )
        self.create_subscription(
            Int32, "camera/quality_factor", self.quality_factor_callback, 10
        )
        self.create_subscription(
            Int32MultiArray, "camera/preview_size", self.preview_size_callback, 10
        )
        
        # ============== SERVICES ==============
        self.create_service(
            GetCameraImage, "get_camera_image", self.get_camera_image_callback
        )
        self.create_service(
            SwitchModel, "switch_ai_model", self.switch_model_callback
        )
        
        # ============== TIMERS ==============
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.status_timer = self.create_timer(1.0, self.status_timer_callback)
        
        self.get_logger().info("OAK Unified Node initialized (DepthAI v3)")
        self._publish_status("idle", "Ready - waiting for subscribers")
    
    def _validate_imu_frequency(self, requested: int) -> int:
        """Find the closest valid BMI270 frequency"""
        return min(BMI270_VALID_FREQUENCIES, key=lambda x: abs(x - requested))
    
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
            self.get_logger().error(
                f"DepthAI v{version} detected. This node REQUIRES DepthAI v3.0.0+. "
                "Please upgrade with: pip install depthai>=3.0.0"
            )
            raise RuntimeError(f"DepthAI v3+ required, found v{version}")
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

        # If we have no pipeline, and we need something, rebuild
        if self.pipeline is None and any(new_config.values()):
            features_changed = True

        # If we have pipeline, check if feature set changed
        elif self.pipeline is not None:
            if new_config != self.current_pipeline_config:
                features_changed = True

            # Also close pipeline if all demands drop to zero (save power/heat)
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
        model_desc = dai.NNModelDescription(slug)
        # Set platform for RVC2 (OAK-D Lite)
        model_desc.platform = "RVC2"
        return model_desc

    def _rebuild_pipeline(self, config):
        """
        Rebuild the pipeline with the current configuration using DepthAI v3 API.
        
        V3 Key differences:
        - dai.node.Camera with .build() instead of ColorCamera
        - camera.requestOutput((w, h)) instead of camera.preview
        - output.createOutputQueue() instead of XLinkOut nodes
        - VideoEncoder.build(output, frameRate=fps, profile=profile)
        - DetectionNetwork.build(cameraNode, modelDescription)
        - pipeline.start()/stop() instead of dai.Device(pipeline)
        """
        with self._pipeline_lock:
            # Stop existing pipeline
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    self.get_logger().warning(f"Error stopping pipeline: {e}")
                self.pipeline = None
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
                # Create new pipeline (NOT as context manager - we need it to persist)
                pipeline = dai.Pipeline()
                
                # --- Camera & Video ---
                needs_camera = config['video'] or config['ai']
                camera_node = None
                camera_output_bgr = None  # For AI (needs BGR)
                camera_output_nv12 = None  # For VideoEncoder (needs NV12)

                if needs_camera:
                    # Create camera with v3 API
                    camera_node = pipeline.create(dai.node.Camera).build(
                        dai.CameraBoardSocket.CAM_A
                    )
                    
                    # Request NV12 output for video encoding (VideoEncoder requires NV12)
                    if config['video']:
                        camera_output_nv12 = camera_node.requestOutput(
                            (self.preview_width, self.preview_height),
                            type=dai.ImgFrame.Type.NV12
                        )
                    
                    # Request BGR output for AI processing
                    if config['ai']:
                        camera_output_bgr = camera_node.requestOutput(
                            (self.preview_width, self.preview_height),
                            type=dai.ImgFrame.Type.BGR888p
                        )

                    # Video Output - MJPEG encoder
                    if config['video']:
                        # Create video encoder with v3 API
                        video_encoder = pipeline.create(dai.node.VideoEncoder).build(
                            camera_output_nv12,
                            frameRate=self.video_fps,
                            profile=dai.VideoEncoderProperties.Profile.MJPEG
                        )
                        # Note: Quality setting may need different approach in v3
                        # video_encoder.setQuality(self.quality_factor)  # May not exist in v3
                        
                        # Create output queue for video
                        self.queues['video'] = video_encoder.out.createOutputQueue(
                            maxSize=4, blocking=False
                        )

                    # AI Output - Detection Network
                    if config['ai']:
                        try:
                            model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
                            node_type = model_info.get('node_type', 'DetectionNetwork')

                            self.get_logger().info(
                                f"Loading model: {self.current_model_name} "
                                f"(type={model_info.get('type')}, node={node_type}) - "
                                "this may take a moment if downloading..."
                            )
                            
                            # Get model description
                            model_desc = self._get_model_description(self.current_model_name)

                            if node_type == 'DetectionNetwork':
                                # v3 API: DetectionNetwork.build(cameraNode, modelDesc)
                                nn = pipeline.create(dai.node.DetectionNetwork).build(
                                    camera_node, model_desc
                                )
                                
                                # Create output queues
                                self.queues['nn'] = nn.out.createOutputQueue(
                                    maxSize=4, blocking=False
                                )
                                # Passthrough for getting frames with detections
                                self.queues['nn_passthrough'] = nn.passthrough.createOutputQueue(
                                    maxSize=4, blocking=False
                                )
                                ai_configured = True
                                
                            elif node_type == 'ParsingNeuralNetwork':
                                # Complex models with custom parsers (depthai-nodes)
                                try:
                                    from depthai_nodes import ParsingNeuralNetwork
                                    nn = ParsingNeuralNetwork.build(camera_node, model_desc)
                                    
                                    self.queues['nn'] = nn.out.createOutputQueue(
                                        maxSize=4, blocking=False
                                    )
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
                                    camera_output_bgr.link(nn.input)
                                    
                                    self.queues['nn'] = nn.out.createOutputQueue(
                                        maxSize=4, blocking=False
                                    )
                                    ai_configured = True
                            else:
                                # Generic NeuralNetwork fallback
                                nn = pipeline.create(dai.node.NeuralNetwork)
                                nn.setNNModelDescription(model_desc)
                                nn.input.setBlocking(False)
                                camera_output_bgr.link(nn.input)
                                
                                self.queues['nn'] = nn.out.createOutputQueue(
                                    maxSize=4, blocking=False
                                )
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
                    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, self.imu_actual_freq)
                    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, self.imu_actual_freq)
                    imu.setBatchReportThreshold(1)
                    imu.setMaxBatchReports(10)

                    self.queues['imu'] = imu.out.createOutputQueue(
                        maxSize=50, blocking=False
                    )

                # Start the pipeline
                pipeline.start()
                self.pipeline = pipeline
                
                # Update config to reflect what was actually configured
                actual_config = config.copy()
                if config['ai'] and not ai_configured:
                    actual_config['ai'] = False
                    self.get_logger().warning("AI was requested but failed to configure")
                
                self.current_pipeline_config = actual_config
                self._model_loading = False

                if actual_config['ai']:
                    self._publish_status("ready", f"Model {self.current_model_name} loaded successfully")

                active_features = [k for k, v in actual_config.items() if v]
                self.get_logger().info(f"Pipeline started: {active_features}")
                
            except Exception as e:
                error_msg = f"Failed to start pipeline: {e}"
                self.get_logger().error(error_msg)
                self._model_loading = False
                self._model_load_error = str(e)
                self._publish_status("error", error_msg)
                self.pipeline = None
                self.queues = {}
                self.current_pipeline_config = {'video': False, 'ai': False, 'imu': False}

    def timer_callback(self):
        """Main processing loop - check demand and process queues"""
        # 1. Check demand and rebuild if necessary
        try:
            self.check_demand()
        except Exception as e:
            self.get_logger().error(f"Error in check_demand: {e}")

        if not self.pipeline:
            return

        # 2. Process MJPEG video frames
        if 'video' in self.queues:
            try:
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
            except Exception as e:
                self.get_logger().error(f"Error processing video: {e}")

        # 3. Process AI
        if 'nn' in self.queues:
            try:
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
            except Exception as e:
                self.get_logger().error(f"Error processing AI: {e}")
                
        # 4. Process IMU
        if 'imu' in self.queues:
            try:
                q = self.queues['imu']
                while q.has():
                    imu_data = q.get()
                    imu_packets = imu_data.packets
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
            except Exception as e:
                self.get_logger().error(f"Error processing IMU: {e}")

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
                
                # Add keypoints if present
                if hasattr(det, 'keypoints') and det.keypoints:
                    det_dict["keypoints"] = []
                    for kp in det.keypoints:
                        det_dict["keypoints"].append({
                            "x": round(kp.x, 4),
                            "y": round(kp.y, 4),
                            "confidence": round(kp.confidence, 4) if hasattr(kp, 'confidence') else 1.0
                        })
                
                # Add mask if present and requested
                if hasattr(det, 'mask') and det.mask is not None:
                    if self.segmentation_mode == "mask":
                        # RLE encode for efficiency
                        mask_array = np.array(det.mask)
                        det_dict["mask_rle"] = rle_encode(mask_array)
                    # Always include mask presence indicator
                    det_dict["has_mask"] = True
                
                detections.append(det_dict)
        except Exception as e:
            return {"error": str(e)}
        
        return {"detections": detections, "count": len(detections)}
    
    def _format_keypoints(self, in_data, model_info: dict) -> dict:
        """Format Keypoints output"""
        try:
            keypoints_list = []
            for kp in in_data.keypoints:
                keypoints_list.append({
                    "x": round(kp.x, 4),
                    "y": round(kp.y, 4),
                    "confidence": round(kp.confidence, 4) if hasattr(kp, 'confidence') else 1.0
                })
            return {"keypoints": keypoints_list, "count": len(keypoints_list)}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_lines(self, in_data) -> dict:
        """Format Lines output"""
        try:
            lines_list = []
            for line in in_data.lines:
                lines_list.append({
                    "start": {"x": round(line.start_x, 4), "y": round(line.start_y, 4)},
                    "end": {"x": round(line.end_x, 4), "y": round(line.end_y, 4)},
                    "confidence": round(line.confidence, 4) if hasattr(line, 'confidence') else 1.0
                })
            return {"lines": lines_list, "count": len(lines_list)}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_predictions(self, in_data, model_type: str) -> dict:
        """Format Predictions output"""
        try:
            predictions = []
            for pred in in_data.predictions:
                predictions.append({
                    "class": pred.label,
                    "confidence": round(pred.confidence, 4)
                })
            return {"predictions": predictions, "count": len(predictions)}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_pose_raw(self, in_data, model_info: dict) -> dict:
        """Format raw pose estimation output"""
        try:
            # Raw NNData handling
            layers = in_data.getAllLayerNames()
            return {"raw_layers": layers, "note": "Install depthai-nodes for parsed output"}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_instance_seg_raw(self, in_data, model_info: dict) -> dict:
        """Format raw instance segmentation output"""
        try:
            layers = in_data.getAllLayerNames()
            return {"raw_layers": layers, "note": "Install depthai-nodes for parsed output"}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_hand_raw(self, in_data, model_info: dict) -> dict:
        """Format raw hand detection output"""
        try:
            if isinstance(in_data, dai.ImgDetections):
                return self._format_detections(in_data)
            layers = in_data.getAllLayerNames()
            return {"raw_layers": layers, "note": "Install depthai-nodes for parsed output"}
        except Exception as e:
            return {"error": str(e)}

    # ============== CALLBACKS ==============
    
    def get_camera_image_callback(self, request, response):
        """Service to get current camera image"""
        response.image_base64 = self.current_image if self.current_image else ""
        return response
    
    def switch_model_callback(self, request, response):
        """Service to switch AI model"""
        model_name = request.model_name
        
        if model_name not in AVAILABLE_MODELS:
            response.success = False
            response.message = f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
            return response
        
        if model_name == self.current_model_name:
            response.success = True
            response.message = f"Already using {model_name}"
            return response
        
        self.get_logger().info(f"Switching model: {self.current_model_name} -> {model_name}")
        self.current_model_name = model_name
        self._force_rebuild = True
        self._publish_status("loading", f"Switching to model {model_name}...")
        self.check_demand()
        
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
        """Publish current model and available models info"""
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
    
    def destroy_node(self):
        """Clean up resources on shutdown"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OakUnifiedNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution failed: {e}")
        import traceback
        traceback.print_exc()
        error_node = ErrorPublisher()
        rclpy.spin(error_node)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
