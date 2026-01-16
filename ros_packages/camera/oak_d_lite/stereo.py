#!/usr/bin/python3
"""
OAK-D Lite Unified Node - Full Integration

Features:
- MJPEG Video Encoding - hardware accelerated (use CBOR subscription for binary transfer)
- AI Inference (MobileNet, YOLO, segmentation, pose) - on-demand activation
- IMU Data (BMI270 6-axis) - accelerometer + gyroscope streaming
- Dynamic Model Selection - switch models at runtime
- RLE-encoded segmentation masks for efficient transmission

All features are on-demand: only active when subscribed.
"""
import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import depthai as dai
import rclpy
import blobconverter
import numpy as np

from datatypes.srv import GetCameraImage, SwitchModel
from rclpy.node import Node
from std_msgs.msg import String, Float64, Int32, Int32MultiArray
from sensor_msgs.msg import Imu, CompressedImage
from geometry_msgs.msg import Vector3Stamped

# ============== MODEL REGISTRY ==============
# Local model paths with fallback to blobconverter
# Models directory on robot (create if using local blobs)
MODELS_DIR = Path("/opt/oak_models/")

AVAILABLE_MODELS = {
    # ============== DETECTION MODELS ==============
    "mobilenet-ssd": {
        "type": "detection",
        "blob": "mobilenet-ssd.blob",
        "zoo_name": "mobilenet-ssd",
        "shaves": 6,
        "description": "Fast object detection (MobileNet SSD)",
        "classes": 20,  # VOC classes
    },
    "yolo-v4-tiny": {
        "type": "detection",
        "blob": "yolo-v4-tiny.blob",
        "zoo_name": "yolo-v4-tiny-tf",
        "shaves": 6,
        "description": "YOLOv4 Tiny - balanced speed/accuracy",
        "classes": 80,  # COCO classes
    },
    "tiny-yolo-v3": {
        "type": "detection",
        "blob": "tiny-yolo-v3.blob",
        "zoo_name": "tiny-yolo-v3",
        "shaves": 6,
        "description": "Tiny YOLOv3 - fast detection",
        "classes": 80,
    },
    "yolov6n": {
        "type": "detection",
        "blob": "yolov6n.blob",
        "zoo_name": "yolov6n_coco_640x640",
        "shaves": 6,
        "description": "YOLOv6 Nano - fast & accurate",
        "classes": 80,
        "input_size": (640, 640),
    },
    "yolov8n": {
        "type": "detection",
        "blob": "yolov8n.blob",
        "zoo_name": "yolov8n_coco_640x640",
        "shaves": 6,
        "description": "YOLOv8 Nano - latest YOLO",
        "classes": 80,
        "input_size": (640, 640),
    },
    "face-detection-retail-0004": {
        "type": "detection",
        "blob": "face-detection-retail-0004.blob",
        "zoo_name": "face-detection-retail-0004",
        "shaves": 6,
        "description": "Face detection model",
        "classes": 1,
    },
    
    # ============== CLASSIFICATION MODELS ==============
    "resnet50": {
        "type": "classification",
        "blob": "resnet50.blob",
        "zoo_name": "resnet-50-pytorch",
        "shaves": 6,
        "description": "ResNet-50 image classification",
        "classes": 1000,  # ImageNet classes
    },
    
    # ============== AGE/GENDER/EMOTION MODELS ==============
    "age-gender": {
        "type": "age-gender",
        "blob": "age-gender.blob",
        "zoo_name": "age-gender-recognition-retail-0013",
        "shaves": 6,
        "description": "Age and gender estimation",
        "outputs": ["age", "gender"],
    },
    "emotion-recognition": {
        "type": "emotion",
        "blob": "emotion-recognition.blob",
        "zoo_name": "emotions-recognition-retail-0003",
        "shaves": 6,
        "description": "Facial emotion recognition",
        "classes": 5,  # neutral, happy, sad, surprise, anger
    },
    
    # ============== SEGMENTATION MODELS ==============
    "deeplabv3": {
        "type": "segmentation",
        "blob": "deeplabv3.blob",
        "zoo_name": "deeplab_v3_plus_mvv2_decoder_256",
        "shaves": 6,
        "description": "DeepLabV3+ multi-class segmentation",
        "classes": 21,  # Pascal VOC classes
        "input_size": (256, 256),
    },
    "deeplabv3-person": {
        "type": "segmentation",
        "blob": "deeplabv3-person.blob",
        "zoo_name": "deeplabv3p_person",
        "shaves": 6,
        "description": "DeepLabV3+ person segmentation (binary)",
        "classes": 2,  # Background + Person
        "input_size": (256, 256),
    },
    "selfie-segmentation": {
        "type": "segmentation",
        "blob": "selfie-segmentation.blob",
        "zoo_name": "mediapipe_selfie",
        "shaves": 6,
        "description": "MediaPipe selfie/portrait segmentation",
        "classes": 2,  # Background + Person
        "input_size": (256, 256),
    },
    
    # ============== INSTANCE SEGMENTATION ==============
    "yolov8n-seg": {
        "type": "instance-segmentation",
        "blob": "yolov8n-seg.blob",
        "zoo_name": "yolov8n-seg",
        "shaves": 6,
        "description": "YOLOv8 Nano instance segmentation",
        "classes": 80,
        "input_size": (640, 640),
    },
    
    # ============== POSE ESTIMATION MODELS ==============
    "human-pose-estimation": {
        "type": "pose",
        "blob": "human-pose-estimation.blob",
        "zoo_name": "human-pose-estimation-0001",
        "shaves": 6,
        "description": "Human pose estimation (18 keypoints)",
        "keypoints": 18,
        "input_size": (456, 256),
    },
    "openpose": {
        "type": "pose",
        "blob": "openpose.blob",
        "zoo_name": "openpose-pose",
        "shaves": 6,
        "description": "OpenPose body keypoint detection",
        "keypoints": 18,
        "input_size": (368, 368),
    },
    "yolov8n-pose": {
        "type": "pose",
        "blob": "yolov8n-pose.blob",
        "zoo_name": "yolov8n-pose",
        "shaves": 6,
        "description": "YOLOv8 Nano pose estimation",
        "keypoints": 17,
        "input_size": (640, 640),
    },
}

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
        self.current_model_name = "mobilenet-ssd"
        self.ai_confidence = 0.5
        self.models_dir = MODELS_DIR
        
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
        
        self.get_logger().info("OakUnifiedNode initialized (MJPEG + AI + IMU). Waiting for subscribers...")

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

        if features_changed:
            self.get_logger().info(f"Demand changed: {new_config}. Rebuilding pipeline...")
            self._rebuild_pipeline(new_config)

    def _rebuild_pipeline(self, config):
        if self.device:
            self.device.close()
            self.device = None
            self.queues = {}

        if not any(config.values()):
            self.current_pipeline_config = config
            return

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

            # AI Output
            if config['ai']:
                try:
                    model_path = self._get_model_path(self.current_model_name)
                    model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
                    model_type = model_info.get('type', 'detection')

                    self.get_logger().info(f"Loading model: {self.current_model_name} (type={model_type})")

                    if model_type == 'detection':
                        nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
                        nn.setConfidenceThreshold(self.ai_confidence)
                    else:
                        # Classification, segmentation, pose, etc.
                        nn = pipeline.create(dai.node.NeuralNetwork)

                    nn.setBlobPath(str(model_path))
                    nn.input.setBlocking(False)
                    camRgb.preview.link(nn.input)

                    xoutNn = pipeline.create(dai.node.XLinkOut)
                    xoutNn.setStreamName("nn")
                    nn.out.link(xoutNn.input)

                except Exception as e:
                    self.get_logger().error(f"Failed to load model {self.current_model_name}: {e}")

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

        try:
            self.device = dai.Device(pipeline)
            self.current_pipeline_config = config

            # Setup queues
            if config['video']:
                self.queues['video'] = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
            if config['ai']:
                self.queues['nn'] = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            if config['imu']:
                self.queues['imu'] = self.device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            active_features = [k for k, v in config.items() if v]
            self.get_logger().info(f"Pipeline started: {active_features}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to start pipeline: {e}")
            self.device = None

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

        # 4. Process AI
        if 'nn' in self.queues:
            q = self.queues['nn']
            while q.has():
                self._frame_count += 1

                in_data = q.get()
                model_info = AVAILABLE_MODELS.get(self.current_model_name, {})
                model_type = model_info.get('type', 'detection')

                # Get sequence number and timestamp from NN output for latency calculation
                # Note: Actual VPU inference time is not directly available; we measure
                # the latency from frame capture to result availability
                try:
                    # DepthAI provides timestamp when the frame was captured
                    frame_ts = in_data.getTimestamp()
                    now = dai.Clock.now()
                    latency_ms = (now - frame_ts).total_seconds() * 1000
                except Exception:
                    latency_ms = 0.0  # Fallback if timestamp not available

                result = self._format_ai_result(in_data, model_type, model_info)

                # Build enhanced output
                output = {
                    "model": self.current_model_name,
                    "type": model_type,
                    "frame_id": self._frame_count,
                    "timestamp_ns": self.get_clock().now().nanoseconds,
                    "latency_ms": round(latency_ms, 2),  # Time from capture to result
                    "result": result
                }
                
                msg = String()
                msg.data = json.dumps(output)
                self.ai_pub.publish(msg)
                
        # 5. Process IMU
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

    def _format_ai_result(self, in_data, model_type: str, model_info: dict) -> dict:
        """Format AI inference result based on model type"""
        if model_type == 'detection':
            detections = []
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
            return {"detections": detections, "count": len(detections)}
        
        elif model_type == 'classification':
            # Get raw output and find top classes
            try:
                data = in_data.getFirstLayerFp16()
                indexed = list(enumerate(data))
                indexed.sort(key=lambda x: x[1], reverse=True)
                
                top_k = []
                for idx, score in indexed[:5]:
                    top_k.append({
                        "class_id": idx,
                        "confidence": round(float(score), 4)
                    })
                return {"classifications": top_k}
            except Exception as e:
                return {"error": str(e)}
        
        elif model_type == 'segmentation':
            # Segmentation with two modes:
            # 1. "bbox" - bounding boxes for all detected classes (lightweight)
            # 2. "mask" - full binary mask for a specific target class (detailed)
            try:
                layer_data = in_data.getFirstLayerFp16()
                mask = np.array(layer_data)
                
                # Handle different tensor formats
                if len(mask.shape) == 3:  # (C, H, W) format
                    class_mask = np.argmax(mask, axis=0)
                    num_classes_in_output = mask.shape[0]
                elif len(mask.shape) == 2:  # (H, W) format - already class indices
                    class_mask = mask.astype(np.int32)
                    num_classes_in_output = int(np.max(class_mask)) + 1
                else:
                    return {"error": f"Unexpected mask shape: {mask.shape}"}
                
                h, w = class_mask.shape
                unique_classes = np.unique(class_mask).tolist()
                
                result = {
                    "mode": self.segmentation_mode,
                    "image_size": [w, h],
                    "classes_detected": unique_classes,
                    "num_classes": model_info.get('classes', num_classes_in_output),
                }
                
                if self.segmentation_mode == "bbox":
                    # Extract bounding boxes for each detected class
                    bboxes = []
                    for class_id in unique_classes:
                        if class_id == 0:  # Skip background
                            continue
                        # Find pixels belonging to this class
                        class_pixels = (class_mask == class_id)
                        if not np.any(class_pixels):
                            continue
                        
                        # Get bounding box
                        rows = np.any(class_pixels, axis=1)
                        cols = np.any(class_pixels, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        
                        # Normalize to 0-1 range
                        bboxes.append({
                            "class_id": int(class_id),
                            "bbox": {
                                "xmin": round(xmin / w, 4),
                                "ymin": round(ymin / h, 4),
                                "xmax": round((xmax + 1) / w, 4),
                                "ymax": round((ymax + 1) / h, 4),
                            },
                            "pixel_count": int(np.sum(class_pixels)),
                            "coverage": round(np.sum(class_pixels) / (h * w), 4),
                        })
                    
                    result["bboxes"] = bboxes
                    result["count"] = len(bboxes)
                    
                elif self.segmentation_mode == "mask" and self.segmentation_target_class is not None:
                    # Return binary mask for the target class using RLE encoding
                    target_class = self.segmentation_target_class
                    binary_mask = (class_mask == target_class).astype(np.uint8)

                    # RLE encode the mask (much smaller than PNG for sparse masks)
                    mask_rle = rle_encode(binary_mask)

                    # Get bbox for the target class
                    class_pixels = binary_mask > 0
                    if np.any(class_pixels):
                        rows = np.any(class_pixels, axis=1)
                        cols = np.any(class_pixels, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        target_bbox = {
                            "xmin": round(xmin / w, 4),
                            "ymin": round(ymin / h, 4),
                            "xmax": round((xmax + 1) / w, 4),
                            "ymax": round((ymax + 1) / h, 4),
                        }
                    else:
                        target_bbox = None

                    result["target_class"] = target_class
                    result["target_bbox"] = target_bbox
                    result["mask_rle"] = mask_rle  # RLE encoded mask (COCO compatible)
                    result["pixel_count"] = int(np.sum(class_pixels))
                
                return result
                
            except Exception as e:
                return {"error": str(e)}
        
        elif model_type == 'pose':
            # Return pose keypoint data
            try:
                # Pose models typically output heatmaps for keypoints
                layer_data = in_data.getFirstLayerFp16()
                num_keypoints = model_info.get('keypoints', 18)
                
                # Try to extract keypoint positions from heatmaps
                keypoints = []
                if hasattr(layer_data, 'shape'):
                    heatmaps = np.array(layer_data)
                    
                    # Typical shape: (num_keypoints, height, width)
                    if len(heatmaps.shape) == 3:
                        for i in range(min(heatmaps.shape[0], num_keypoints)):
                            heatmap = heatmaps[i]
                            # Find peak location
                            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                            confidence = float(heatmap[max_idx])
                            # Normalize to 0-1 range
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
                }
            except Exception as e:
                return {"error": str(e)}
        
        elif model_type == 'age-gender':
            # Age and gender estimation output
            try:
                # These models typically have two output layers
                age_data = in_data.getLayerFp16("age_conv3") if in_data.getLayerNames() else in_data.getFirstLayerFp16()
                gender_data = in_data.getLayerFp16("prob") if "prob" in in_data.getLayerNames() else None
                
                # Age is typically a single float (normalized 0-1, multiply by 100 for years)
                age = float(age_data[0]) * 100 if len(age_data) > 0 else None
                
                # Gender is typically [female_prob, male_prob]
                if gender_data is not None and len(gender_data) >= 2:
                    female_prob = float(gender_data[0])
                    male_prob = float(gender_data[1])
                    gender = "male" if male_prob > female_prob else "female"
                    gender_confidence = max(male_prob, female_prob)
                else:
                    gender = None
                    gender_confidence = 0
                
                return {
                    "age": round(age, 1) if age else None,
                    "gender": gender,
                    "gender_confidence": round(gender_confidence, 4),
                }
            except Exception as e:
                return {"error": str(e)}
        
        elif model_type == 'emotion':
            # Emotion recognition output
            try:
                data = in_data.getFirstLayerFp16()
                emotions = ["neutral", "happy", "sad", "surprise", "anger"]
                
                # Find top emotion
                probs = list(data[:len(emotions)])
                max_idx = np.argmax(probs)
                
                return {
                    "emotion": emotions[max_idx],
                    "confidence": round(float(probs[max_idx]), 4),
                    "all_emotions": {emotions[i]: round(float(probs[i]), 4) for i in range(len(emotions))},
                }
            except Exception as e:
                return {"error": str(e)}
        
        elif model_type == 'instance-segmentation':
            # Instance segmentation (e.g., YOLOv8-seg) - detection + per-instance masks
            try:
                # YOLOv8-seg outputs detections + mask coefficients
                # For now, return detections similar to regular detection
                detections = []
                if hasattr(in_data, 'detections'):
                    for det in in_data.detections:
                        detections.append({
                            "label": det.label,
                            "confidence": round(det.confidence, 4),
                            "bbox": {
                                "xmin": round(det.xmin, 4),
                                "ymin": round(det.ymin, 4),
                                "xmax": round(det.xmax, 4),
                                "ymax": round(det.ymax, 4),
                            },
                        })
                
                return {
                    "detections": detections,
                    "count": len(detections),
                    "note": "Instance masks available via mask mode",
                }
            except Exception as e:
                return {"error": str(e)}
        
        else:
            # Generic output for unknown types
            return {"raw": "unsupported model type"}
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get model path - try local first, then blobconverter"""
        model_info = AVAILABLE_MODELS.get(model_name, {})
        
        # Try local path first
        local_path = self.models_dir / model_info.get('blob', f'{model_name}.blob')
        if local_path.exists():
            self.get_logger().info(f"Using local model: {local_path}")
            return local_path
        
        # Fall back to blobconverter
        zoo_name = model_info.get('zoo_name', model_name)
        shaves = model_info.get('shaves', 6)
        self.get_logger().info(f"Downloading model from zoo: {zoo_name}")
        return Path(blobconverter.from_zoo(name=zoo_name, shaves=shaves))
    
    def _validate_imu_frequency(self, requested_freq: int) -> int:
        """Validate and round IMU frequency to BMI270 supported values"""
        # BMI270 rounds DOWN to nearest valid frequency
        valid = [f for f in BMI270_VALID_FREQUENCIES if f <= requested_freq]
        if not valid:
            return BMI270_VALID_FREQUENCIES[0]  # Minimum
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
        
        self.get_logger().info(f"Switching model to: {model_name}")
        self.current_model_name = model_name
        
        # Trigger rebuild if AI is active
        if self.current_pipeline_config['ai']:
            self.current_pipeline_config['ai'] = False 
            self.check_demand()
             
        response.success = True
        response.message = f"Switched to {model_name}"
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
                    
                    if self.current_pipeline_config['imu']:
                        self.current_pipeline_config['imu'] = False
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
            
            # Segmentation mode change (no rebuild needed, just changes output format)
            if 'segmentation_mode' in data:
                mode = data['segmentation_mode']
                if mode in ['bbox', 'mask']:
                    self.segmentation_mode = mode
                    self.get_logger().info(f"Segmentation mode set to: {mode}")
                else:
                    self.get_logger().error(f"Invalid segmentation_mode: {mode}. Use 'bbox' or 'mask'")
            
            # Segmentation target class (for mask mode)
            if 'segmentation_target_class' in data:
                target = data['segmentation_target_class']
                if target is None or isinstance(target, int):
                    self.segmentation_target_class = target
                    self.get_logger().info(f"Segmentation target class set to: {target}")
                else:
                    self.get_logger().error(f"Invalid segmentation_target_class: {target}. Use integer or null")
                
            if rebuild_needed and self.current_pipeline_config['ai']:
                self.current_pipeline_config['ai'] = False 
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
                    self.quality_factor = max(1, min(100, new_quality))  # Clamp 1-100
                    rebuild_needed = True
                    
            if 'resolution' in config:
                w, h = config['resolution']
                if w != self.preview_width or h != self.preview_height:
                    self.preview_width = w
                    self.preview_height = h
                    rebuild_needed = True
            
            if rebuild_needed:
                self.get_logger().info(f"Camera config updated: fps={self.video_fps}, resolution={self.preview_width}x{self.preview_height}")
                # Rebuild if video feature is active
                if self.current_pipeline_config.get('video'):
                    self._rebuild_pipeline(self.current_pipeline_config)
                    
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
            "classes": model_info.get('classes', 0)
        })
        self.ai_current_pub.publish(msg_curr)
        
        # Publish available models with metadata
        models_list = {}
        for name, info in AVAILABLE_MODELS.items():
            models_list[name] = {
                "type": info.get('type', 'unknown'),
                "description": info.get('description', ''),
                "classes": info.get('classes', 0)
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
        # Rebuild if video or AI is active (both use camera)
        if self.current_pipeline_config.get('video') or self.current_pipeline_config.get('ai'):
            self._rebuild_pipeline(self.current_pipeline_config)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OakUnifiedNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution failed: {e}")
        # Fallback to error publisher or exit
        error_node = ErrorPublisher()
        rclpy.spin(error_node)
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
