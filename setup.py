"""
CNN-Based Damage Change Segmentation Package Setup
==================================================

A comprehensive deep learning package for damage change detection and segmentation
using advanced CNN architectures and transformer-based models.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="damage-change-segmentation",
    version="1.0.0",
    author="Damage Change Assessment Research Team",
    author_email="research@damage-change.org",
    description="A comprehensive deep learning framework for damage change detection and segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IH-Arik/CNN-Based-Damage-Change-Segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "torchvision>=0.13.0+cu116",
        ],
    },
    entry_points={
        "console_scripts": [
            "damage-train=src.training.train:main",
            "damage-eval=src.inference.evaluate:main",
            "damage-predict=src.inference.predict:main",
        ],
    },
)
