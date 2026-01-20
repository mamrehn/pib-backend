#!/usr/bin/env python3
"""
Pre-download OAK-D models from Luxonis Model Hub for offline use.

This script uses DepthAI v3's NNModelDescription API to download and cache
neural network models. The models will be automatically converted to the
appropriate format for the target device (RVC2 for OAK-D Lite).

Usage:
    python download_oak_models.py [--models MODEL1,MODEL2,...] [--all]

Examples:
    python download_oak_models.py --all
    python download_oak_models.py --models yolov8n,yolov6n,face
"""

import argparse
import sys
from pathlib import Path

try:
    import depthai as dai
except ImportError:
    print("ERROR: depthai not installed. Install with: pip install depthai")
    sys.exit(1)

# DepthAI v3 Model Hub slugs for RVC2 devices (OAK-D Lite)
# Format: "luxonis/model-name:variant"
MODEL_SLUGS = {
    # Object Detection
    "yolov8n": "luxonis/yolov8-nano:coco-512x288",
    "yolov6n": "luxonis/yolov6-nano:r2-coco-512x288",
    "yolov10n": "luxonis/yolov10-nano:coco-512x288",
    
    # Face Detection
    "face": "luxonis/scrfd:2.5g-kps-640x640",
    
    # Person Detection
    "person": "luxonis/scrfd-person-detection:25g-640x640",
    
    # Pose Estimation
    "pose_yolo": "luxonis/yolov8-nano-pose-estimation:coco-512x288",
    "pose_hrnet": "luxonis/lite-hrnet:18-coco-288x384",
    
    # Hand Tracking
    "hand": "luxonis/mediapipe-hand-landmarker:224x224",
    
    # Instance Segmentation
    "segmentation": "luxonis/yolov8-instance-segmentation-nano:coco-512x288",
    
    # Gaze Estimation
    "gaze": "luxonis/l2cs-net:448x448",
    
    # Line Detection
    "lines": "luxonis/m-lsd:512x512",
}

# Default models to download (most commonly used)
DEFAULT_MODELS = ["yolov8n", "face", "person", "pose_yolo", "hand", "segmentation"]


def download_model(model_key: str, model_slug: str, verbose: bool = True) -> bool:
    """
    Download a model from Luxonis Model Hub.
    
    The model will be cached locally by DepthAI for future offline use.
    
    Args:
        model_key: Short name for the model
        model_slug: Full Luxonis Model Hub slug
        verbose: Print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Downloading: {model_key}")
        print(f"Model Hub slug: {model_slug}")
        print(f"{'='*60}")
    
    try:
        # Create model description with platform set for RVC2 (OAK-D Lite)
        model_desc = dai.NNModelDescription(model_slug)
        model_desc.platform = "RVC2"  # Platform is a string, not enum
        
        # Get model path (this downloads if not cached)
        # progressFormat can be: 'none', 'bar', 'percent', 'pretty', 'json'
        model_path = dai.getModelFromZoo(model_desc, progressFormat='pretty' if verbose else 'none')
        
        if verbose:
            print(f"✓ Downloaded successfully!")
            print(f"  Cached at: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_key}: {e}")
        return False


def list_available_models():
    """Print all available models."""
    print("\nAvailable models:")
    print("-" * 60)
    
    # Group by category
    categories = {
        "Object Detection": ["yolov8n", "yolov6n", "yolov10n"],
        "Face Detection": ["face"],
        "Person Detection": ["person"],
        "Pose Estimation": ["pose_yolo", "pose_hrnet"],
        "Hand Tracking": ["hand"],
        "Instance Segmentation": ["segmentation"],
        "Gaze Estimation": ["gaze"],
        "Line Detection": ["lines"],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            slug = MODEL_SLUGS.get(key, "unknown")
            default_marker = " (default)" if key in DEFAULT_MODELS else ""
            print(f"  {key:20} -> {slug}{default_marker}")


def check_depthai_version():
    """Check that we have DepthAI v3+."""
    version = getattr(dai, '__version__', '0.0.0')
    major = int(version.split('.')[0])
    
    if major < 3:
        print(f"WARNING: DepthAI version {version} detected.")
        print("         This script requires DepthAI v3.0.0+")
        print("         Install with: pip install depthai>=3.0.0")
        return False
    
    print(f"DepthAI version: {version}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download OAK-D models from Luxonis Model Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                     Download all available models
  %(prog)s --models yolov8n,face     Download specific models
  %(prog)s --list                    List available models
  %(prog)s                           Download default models
        """
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        help="Comma-separated list of models to download"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        list_available_models()
        return 0
    
    # Version check
    print("DepthAI v3 Model Downloader for OAK-D Lite")
    print("=" * 50)
    
    if not check_depthai_version():
        return 1
    
    # Determine which models to download
    if args.all:
        models_to_download = list(MODEL_SLUGS.keys())
        print(f"\nDownloading ALL {len(models_to_download)} models...")
    elif args.models:
        models_to_download = [m.strip() for m in args.models.split(",")]
        # Validate
        invalid = [m for m in models_to_download if m not in MODEL_SLUGS]
        if invalid:
            print(f"ERROR: Unknown models: {invalid}")
            print("Use --list to see available models")
            return 1
        print(f"\nDownloading {len(models_to_download)} specified models...")
    else:
        models_to_download = DEFAULT_MODELS
        print(f"\nDownloading {len(models_to_download)} default models...")
        print("(Use --all for all models, or --models to specify)")
    
    # Download each model
    verbose = not args.quiet
    success_count = 0
    fail_count = 0
    
    for model_key in models_to_download:
        slug = MODEL_SLUGS[model_key]
        if download_model(model_key, slug, verbose=verbose):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Total:      {len(models_to_download)}")
    
    if fail_count > 0:
        print("\nSome downloads failed. Check network connection and try again.")
        return 1
    
    print("\n✓ All models downloaded successfully!")
    print("  Models are cached and will work offline.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
