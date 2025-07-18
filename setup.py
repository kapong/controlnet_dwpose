from setuptools import setup, find_packages

with open("requirement.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dwpose",
    version="0.1.0",
    author="Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, Jiaya Jia",
    author_email="",
    description="DWPose component from ControlNeXt for whole-body pose estimation",
    long_description="DWPose is a whole-body pose estimation library extracted from ControlNeXt project. It provides dense keypoint detection for body, hands, and face using ONNX Runtime. Originally developed as part of the ControlNeXt controllable generation framework.",
    long_description_content_type="text/plain",
    url="https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2/dwpose",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    keywords="pose-estimation controlnext controllable-generation onnx",
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/dvlab-research/ControlNeXt/issues",
        "Source": "https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2/dwpose",
        "Parent Project": "https://github.com/dvlab-research/ControlNeXt",
    },
)