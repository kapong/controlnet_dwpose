# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a DWPose (Dense Whole-body Pose) detection library extracted from the ControlNeXt project. It provides human pose estimation using ONNX models for controllable generation tasks. The main components:

### Core Components

- **DWposeDetector** (`dwpose_detector.py`): Main interface class that coordinates the pose detection pipeline
- **Wholebody** (`wholebody.py`): Core detection engine that combines human detection and pose estimation
- **Detection Pipeline** (`onnxdet.py`): Human detection using YOLOX model with NMS post-processing
- **Pose Estimation** (`onnxpose.py`): RTMPose-based pose estimation with SimCC decoding
- **Preprocessing** (`preprocess.py`): Video/image preprocessing with pose rescaling for temporal consistency
- **Visualization** (`util.py`): Pose rendering utilities for body, hand, and face keypoints

### Key Architecture Patterns

- **Lazy Loading**: Models are loaded only when first called (`DWposeDetector.__call__`)
- **Memory Management**: Explicit memory cleanup with `release_memory()` method
- **ONNX Runtime**: All inference uses ONNX Runtime with CPU/CUDA providers
- **Coordinate System**: Normalized coordinates (0-1) for pose keypoints, converted to pixel coordinates for visualization

### Model Dependencies

The system expects pre-trained ONNX models:
- Detection model: `pretrained/DWPose/yolox_l.onnx` (YOLOX-L for human detection)
- Pose model: `pretrained/DWPose/dw-ll_ucoco_384.onnx` (DWPose whole-body pose estimation)

### Processing Pipeline

1. **Human Detection**: YOLOX detects people in the image
2. **Pose Estimation**: RTMPose estimates keypoints for detected persons
3. **Keypoint Mapping**: Converts between MMPose and OpenPose formats
4. **Visualization**: Renders pose on canvas with confidence-based alpha blending

## Development Notes

- The codebase uses NumPy extensively for array operations
- OpenCV is used for image processing and visualization
- The system supports both CPU and CUDA execution
- Video processing uses decord for efficient frame reading
- Pose rescaling maintains temporal consistency across video frames
- Originally developed as part of the ControlNeXt controllable generation framework

## Dependencies

Core dependencies (based on imports):
- numpy
- opencv-python (cv2)
- onnxruntime-gpu
- torch (for device detection)
- matplotlib (for color mapping)
- decord (for video processing)
- tqdm (for progress bars)

Install dependencies with: `pip install -r requirement.txt`

## Attribution

This code is extracted from the ControlNeXt project by dvlab-research:
- Original repository: https://github.com/dvlab-research/ControlNeXt
- Specific source: https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2/dwpose