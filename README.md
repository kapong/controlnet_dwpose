# DWPose

A standalone Python library for whole-body pose estimation extracted from the [ControlNeXt](https://github.com/dvlab-research/ControlNeXt) project.

## Overview

DWPose provides dense keypoint detection for body, hands, and face using ONNX Runtime. It supports both CPU and GPU inference and is designed for efficient pose estimation in images and videos.

## Features

- **Whole-body pose estimation**: Detects body (18 keypoints), hands (21 keypoints each), and face (68 keypoints)
- **ONNX Runtime support**: Efficient inference with CPU and CUDA providers
- **Video processing**: Temporal consistency with pose rescaling across frames
- **Memory efficient**: Lazy loading and explicit memory management
- **Easy to use**: Simple API with minimal setup

## Installation

### From PyPI (when available)
```bash
pip install dwpose
```

### From source
```bash
git clone https://github.com/your-username/dwpose.git
cd dwpose
pip install -e .
```

### Dependencies
Install required dependencies:
```bash
pip install -r requirement.txt
```

## Quick Start

### Basic Usage

```python
import cv2
import numpy as np
from dwpose import DWposeDetector

# Initialize detector
detector = DWposeDetector(
    model_det="path/to/yolox_l.onnx",
    model_pose="path/to/dw-ll_ucoco_384.onnx",
    device='cpu'  # or 'cuda'
)

# Load image
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect poses
pose_result = detector(image)

# Access results
bodies = pose_result['bodies']
hands = pose_result['hands']
faces = pose_result['faces']

# Clean up memory when done
detector.release_memory()
```

### Processing Images

```python
from dwpose.preprocess import get_image_pose

# Get pose visualization
pose_image = get_image_pose(image)
```

### Processing Videos

```python
from dwpose.preprocess import get_video_pose

# Process video with pose rescaling
pose_sequence = get_video_pose(
    video_path="path/to/video.mp4",
    ref_image=reference_image,
    sample_stride=1
)
```

## Model Requirements

You need to download the ONNX models:

1. **Detection model**: `yolox_l.onnx` - YOLOX-L for human detection
2. **Pose model**: `dw-ll_ucoco_384.onnx` - DWPose for keypoint estimation

Place these models in your preferred directory and update the paths in your code.

## Output Format

The pose detection returns a dictionary with:

- `bodies`: Body keypoints with shape `(N, 18, 2)` for N detected persons
- `hands`: Hand keypoints with shape `(N, 42, 2)` (both hands combined)
- `faces`: Face keypoints with shape `(N, 68, 2)`
- `*_score`: Confidence scores for each keypoint type

Coordinates are normalized (0-1) relative to image dimensions.

## Requirements

- Python >= 3.7
- numpy
- opencv-python
- onnxruntime-gpu (or onnxruntime for CPU-only)
- torch
- matplotlib
- decord
- tqdm

## Attribution

This code is extracted and packaged from the ControlNeXt project:

- **Original Repository**: [dvlab-research/ControlNeXt](https://github.com/dvlab-research/ControlNeXt)
- **Source Directory**: [ControlNeXt-SVD-v2/dwpose](https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2/dwpose)
- **Authors**: Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, Jiaya Jia

### Citation

If you use this code, please cite the original ControlNeXt paper:

```bibtex
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the original [ControlNeXt repository](https://github.com/dvlab-research/ControlNeXt) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please open an issue on the [GitHub repository](https://github.com/your-username/dwpose/issues).