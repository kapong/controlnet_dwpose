# DWPose

[![PyPI version](https://badge.fury.io/py/controlnet-dwpose.svg)](https://badge.fury.io/py/controlnet-dwpose)
[![Build Status](https://github.com/kapong/controlnet_dwpose/workflows/Test%20Package/badge.svg)](https://github.com/kapong/controlnet_dwpose/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/kapong/38fc41f2ca2b2f0fba131d96183a11e4/dwpose.ipynb)

A standalone Python library for whole-body pose estimation extracted from the [ControlNeXt](https://github.com/dvlab-research/ControlNeXt) project.

> **Note**: This is a modified version of the DWPose implementation from the original ControlNeXt repository. The code has been restructured for standalone use and includes API improvements for better usability.

## ⚠️ AI Generation Disclaimer

This repository's packaging, documentation, and API improvements were generated with assistance from Claude AI. While the core DWPose algorithms remain unchanged from the original research implementations, the following components were created using AI assistance:

- Repository structure and packaging (setup.py, requirements.txt)
- Documentation (README.md, CLAUDE.md)
- API improvements and code organization
- Installation and usage examples

The original DWPose research and implementation credit belongs to the respective authors (Yang et al. and Peng et al.). This packaging is provided for educational and research purposes.

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
pip install controlnet_dwpose
```

### From source
```bash
git clone https://github.com/kapong/controlnet_dwpose.git
cd controlnet_dwpose
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
from controlnet_dwpose import DWposeDetector

# Initialize detector
detector = DWposeDetector(
    model_det="yolox_l.onnx",
    model_pose="dw-ll_ucoco_384.onnx",
    device='cpu'  # or 'cuda'
)

# Load image
image = cv2.imread('example/02.jpeg')
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

### Processing Images with Customizable Thickness

```python
from controlnet_dwpose.preprocess import get_image_pose
from controlnet_dwpose import set_thickness_multiplier, get_thickness_multiplier

# Method 1: Using the new API functions (recommended)
current_thickness = get_thickness_multiplier()  # Get current value (default: 3)
set_thickness_multiplier(2)  # Set thinner lines
# set_thickness_multiplier(5)  # Or set thicker lines

# Method 2: Direct access (legacy)
import controlnet_dwpose.util as util
util.thickness_mul = 2  # Thinner lines

# Get pose visualization
pose_image = get_image_pose(detector, image)

# Convert from CHW to HWC format for display
pose_image = pose_image.transpose(1, 2, 0)

# Display or save
import matplotlib.pyplot as plt
plt.imshow(pose_image)
plt.axis('off')
plt.show()
```

### Processing Videos

```python
from controlnet_dwpose.preprocess import get_video_pose
from controlnet_dwpose import set_thickness_multiplier

# Configure thickness for video processing
set_thickness_multiplier(2)

# Process video with pose rescaling
pose_sequence = get_video_pose(
    dwprocessor=detector,
    video_path="path/to/video.mp4",
    ref_image=reference_image,
    sample_stride=1
)
```

### Advanced Usage with Custom Visualization

```python
from controlnet_dwpose.util import draw_pose
from controlnet_dwpose import set_thickness_multiplier

# Set custom thickness
set_thickness_multiplier(4)

# Get pose data
pose_result = detector(image)
height, width = image.shape[:2]

# Create custom visualization
pose_canvas = draw_pose(pose_result, height, width)

# Convert to displayable format
pose_image = pose_canvas.transpose(1, 2, 0)  # CHW to HWC
```

## Thickness Configuration

The library provides configurable thickness for pose visualization through dedicated API functions:

### API Functions

```python
from controlnet_dwpose import set_thickness_multiplier, get_thickness_multiplier

# Get current thickness multiplier (default: 3)
current_thickness = get_thickness_multiplier()
print(f"Current thickness: {current_thickness}")

# Set new thickness multiplier
set_thickness_multiplier(2)    # Thinner lines
set_thickness_multiplier(5)    # Thicker lines
set_thickness_multiplier(1.5)  # Fine control with float values

# Verify the change
new_thickness = get_thickness_multiplier()
print(f"New thickness: {new_thickness}")
```

### Effects on Visualization

The thickness multiplier affects:
- **Body pose lines**: Line width = 4 × thickness_multiplier
- **Body keypoints**: Circle radius = 4 × thickness_multiplier  
- **Hand pose lines**: Line width = 2 × thickness_multiplier
- **Hand keypoints**: Circle radius = 4 × thickness_multiplier
- **Face keypoints**: Circle radius = 3 × thickness_multiplier

### Legacy Access

For backward compatibility, direct access is still supported:

```python
import controlnet_dwpose.util as util
util.thickness_mul = 2  # Direct assignment
```

## Model Requirements

You need to download the ONNX models:

1. **Detection model**: `yolox_l.onnx` - YOLOX-L for human detection
2. **Pose model**: `dw-ll_ucoco_384.onnx` - DWPose for keypoint estimation

### Download Models

Install gdown for downloading from Google Drive:
```bash
pip install gdown
```

Download the required models:
```python
import gdown

# Download pose estimation model
gdown.download('https://drive.google.com/uc?id=12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2', 'dw-ll_ucoco_384.onnx', quiet=False)

# Download detection model  
gdown.download('https://drive.google.com/uc?id=1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI', 'yolox_l.onnx', quiet=False)
```

Or from command line:
```bash
gdown 'https://drive.google.com/uc?id=12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2' -O dw-ll_ucoco_384.onnx
gdown 'https://drive.google.com/uc?id=1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI' -O yolox_l.onnx
```

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

- **ControlNeXt Repository**: [dvlab-research/ControlNeXt](https://github.com/dvlab-research/ControlNeXt)
- **Source Directory**: [ControlNeXt-SVD-v2/dwpose](https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2/dwpose)
- **ControlNeXt Authors**: Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, Jiaya Jia

### Original DWPose

The DWPose implementation used by Peng et al. in ControlNeXt is based on the original DWPose research:

- **Original DWPose Repository**: [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)
- **Original DWPose Authors**: Zhendong Yang, Ailing Zeng, Chun Yuan, Yu Li
- **Paper**: "Effective Whole-body Pose Estimation with Two-stages Distillation" (ICCV 2023)

### Modifications

This repository contains the following modifications from the original ControlNeXt implementation:

- **Standalone packaging**: Restructured as an independent Python package with proper setup.py
- **API improvements**: Enhanced function signatures for better usability
- **Documentation**: Added comprehensive README, examples, and installation instructions
- **Dependency management**: Proper requirements.txt and package dependencies
- **Model download**: Integrated Google Drive download links for pre-trained models
- **Memory management**: Improved memory cleanup and resource handling

### Citation

If you use this code, please cite both the ControlNeXt paper and the original DWPose paper:

```bibtex
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}

@inproceedings{yang2023effective,
  title={Effective Whole-body Pose Estimation with Two-stages Distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4210--4219},
  year={2023}
}
```

## License

This project is licensed under the Apache License 2.0 - see the original [ControlNeXt repository](https://github.com/dvlab-research/ControlNeXt) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please open an issue on the [GitHub repository](https://github.com/your-username/dwpose/issues).