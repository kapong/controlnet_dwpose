"""
DWPose: Dense Whole-body Pose Estimation

A Python library for whole-body pose estimation using ONNX models.
Extracted from the ControlNeXt project for standalone use.
"""

from .dwpose_detector import DWposeDetector
from .wholebody import Wholebody
from .preprocess import get_image_pose, get_video_pose
from .util import draw_pose, draw_bodypose, draw_handpose, draw_facepose

__version__ = "0.1.3"
__author__ = "Phongphan Phienphanich"
__email__ = "garpong@gmail.com"
__license__ = "Apache-2.0"

__all__ = [
    "DWposeDetector",
    "Wholebody",
    "get_image_pose",
    "get_video_pose",
    "draw_pose",
    "draw_bodypose",
    "draw_handpose",
    "draw_facepose",
]