"""
Setup script for waste_classifier package.

This allows installing the package in editable mode:
    pip install -e .

Author: Pham An
"""

from setuptools import setup, find_packages

setup(
    name="waste_classifier",
    version="2.0.0",
    author="Pham An",
    description="Production-ready waste classification system using deep learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/phaman/waste_classifier_capstone",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

