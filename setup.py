from setuptools import setup, find_packages
import os

# Read requirements file if it exists
requirements = []
if os.path.exists("requirement.txt"):
    with open("requirement.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    # Fallback to hardcoded requirements
    requirements = [
        "numpy",
        "opencv-python",
        "onnxruntime",
        "torch",
        "matplotlib",
        "decord",
        "tqdm",
    ]

setup(
    name="controlnet_dwpose",
    version="0.1.4",
    author="P.Phienphanich",
    author_email="garpong@gmail.com",
    description="DWPose component from ControlNeXt for whole-body pose estimation",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "DWPose is a whole-body pose estimation library extracted and packaged from ControlNeXt project by Peng et al.",
    long_description_content_type="text/markdown",
    url="https://github.com/kapong/controlnet_dwpose",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    keywords="pose-estimation controlnext controllable-generation onnx",
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/kapong/controlnet_dwpose/issues",
        "Source": "https://github.com/kapong/controlnet_dwpose",
        "Original ControlNeXt": "https://github.com/dvlab-research/ControlNeXt",
        "Original DWPose": "https://github.com/IDEA-Research/DWPose",
    },
)