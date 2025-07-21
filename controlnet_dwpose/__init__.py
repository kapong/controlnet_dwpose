"""
DWPose: Dense Whole-body Pose Estimation

A Python library for whole-body pose estimation using ONNX models.
Extracted from the ControlNeXt project for standalone use.
"""

from .dwpose_detector import DWposeDetector
from .wholebody import Wholebody
from .preprocess import get_image_pose, get_video_pose
from .util import draw_pose, draw_bodypose, draw_handpose, draw_facepose, set_thickness_multiplier, get_thickness_multiplier

__all__ = [
    "DWposeDetector",
    "Wholebody",
    "get_image_pose",
    "get_video_pose",
    "draw_pose",
    "draw_bodypose",
    "draw_handpose",
    "draw_facepose",
    "set_thickness_multiplier",
    "get_thickness_multiplier",
]