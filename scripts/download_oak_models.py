#!/usr/bin/env python3
"""
Pre-download all OAK-D Lite AI models for offline use.

This script downloads all models from the DepthAI model zoo and saves them
to /opt/oak_models/ (or a custom directory). Models are cached by blobconverter,
so running this script multiple times won't re-download existing models.

Usage:
    python3 download_oak_models.py                    # Download to /opt/oak_models/
    python3 download_oak_models.py --output ./models  # Custom output directory
    python3 download_oak_models.py --list             # List available models
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import blobconverter
except ImportError:
    print("Error: blobconverter not installed. Run: pip install blobconverter")
    sys.exit(1)


# Model registry - must match stereo.py
AVAILABLE_MODELS = {
    # ============== DETECTION MODELS ==============
    "mobilenet-ssd": {
        "zoo_name": "mobilenet-ssd",
        "shaves": 6,
        "description": "Fast object detection (MobileNet SSD)",
    },
    "yolo-v4-tiny": {
        "zoo_name": "yolo-v4-tiny-tf",
        "shaves": 6,
        "description": "YOLOv4 Tiny - balanced speed/accuracy",
    },
    "tiny-yolo-v3": {
        "zoo_name": "tiny-yolo-v3",
        "shaves": 6,
        "description": "Tiny YOLOv3 - fast detection",
    },
    "yolov6n": {
        "zoo_name": "yolov6n_coco_640x640",
        "shaves": 6,
        "description": "YOLOv6 Nano - fast & accurate",
    },
    "yolov8n": {
        "zoo_name": "yolov8n_coco_640x640",
        "shaves": 6,
        "description": "YOLOv8 Nano - latest YOLO",
    },
    "face-detection-retail-0004": {
        "zoo_name": "face-detection-retail-0004",
        "shaves": 6,
        "description": "Face detection model",
    },
    
    # ============== CLASSIFICATION MODELS ==============
    "resnet50": {
        "zoo_name": "resnet-50-pytorch",
        "shaves": 6,
        "description": "ResNet-50 image classification",
    },
    
    # ============== AGE/GENDER/EMOTION MODELS ==============
    "age-gender": {
        "zoo_name": "age-gender-recognition-retail-0013",
        "shaves": 6,
        "description": "Age and gender estimation",
    },
    "emotion-recognition": {
        "zoo_name": "emotions-recognition-retail-0003",
        "shaves": 6,
        "description": "Facial emotion recognition",
    },
    
    # ============== SEGMENTATION MODELS ==============
    "deeplabv3": {
        "zoo_name": "deeplab_v3_plus_mvv2_decoder_256",
        "shaves": 6,
        "description": "DeepLabV3+ multi-class segmentation",
    },
    "deeplabv3-person": {
        "zoo_name": "deeplabv3p_person",
        "shaves": 6,
        "description": "DeepLabV3+ person segmentation",
    },
    "selfie-segmentation": {
        "zoo_name": "mediapipe_selfie",
        "shaves": 6,
        "description": "MediaPipe selfie/portrait segmentation",
    },
    
    # ============== INSTANCE SEGMENTATION ==============
    "yolov8n-seg": {
        "zoo_name": "yolov8n-seg",
        "shaves": 6,
        "description": "YOLOv8 Nano instance segmentation",
    },
    
    # ============== POSE ESTIMATION MODELS ==============
    "human-pose-estimation": {
        "zoo_name": "human-pose-estimation-0001",
        "shaves": 6,
        "description": "Human pose estimation (18 keypoints)",
    },
    "openpose": {
        "zoo_name": "openpose-pose",
        "shaves": 6,
        "description": "OpenPose body keypoint detection",
    },
    "yolov8n-pose": {
        "zoo_name": "yolov8n-pose",
        "shaves": 6,
        "description": "YOLOv8 Nano pose estimation",
    },
}


def download_model(name: str, info: dict, output_dir: Path) -> bool:
    """Download a single model and copy to output directory."""
    zoo_name = info["zoo_name"]
    shaves = info["shaves"]
    output_path = output_dir / f"{name}.blob"
    
    if output_path.exists():
        print(f"  ✓ {name}: Already exists")
        return True
    
    try:
        print(f"  ↓ {name}: Downloading {zoo_name}...", end=" ", flush=True)
        blob_path = blobconverter.from_zoo(name=zoo_name, shaves=shaves)
        
        # Copy to output directory with our naming convention
        shutil.copy(blob_path, output_path)
        print(f"OK ({output_path.stat().st_size // 1024} KB)")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():

    # Simplest usage: sudo python3 scripts/download_oak_models.py
    # Downloads all models to /opt/oak_models/ by default.

    parser = argparse.ArgumentParser(
        description="Pre-download OAK-D Lite AI models for offline use"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("/opt/oak_models"),
        help="Output directory for model files (default: /opt/oak_models)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Download specific models only (space-separated names)"
    )
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("Available models:\n")
        for name, info in AVAILABLE_MODELS.items():
            print(f"  {name:30} - {info['description']}")
        print(f"\nTotal: {len(AVAILABLE_MODELS)} models")
        return 0
    
    # Determine which models to download
    if args.models:
        models_to_download = {
            name: info for name, info in AVAILABLE_MODELS.items()
            if name in args.models
        }
        unknown = set(args.models) - set(AVAILABLE_MODELS.keys())
        if unknown:
            print(f"Warning: Unknown models ignored: {unknown}")
    else:
        models_to_download = AVAILABLE_MODELS
    
    # Create output directory
    output_dir = args.output
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create {output_dir}. Try with sudo or use --output to specify a different directory.")
        return 1
    
    print(f"Downloading {len(models_to_download)} models to {output_dir}\n")
    
    # Download models
    success = 0
    failed = 0
    
    for name, info in models_to_download.items():
        if download_model(name, info, output_dir):
            success += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Downloaded: {success}/{len(models_to_download)} models")
    if failed:
        print(f"Failed: {failed} models")
        return 1
    
    print(f"\nModels saved to: {output_dir}")
    print("These will be used automatically by the OakUnifiedNode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
