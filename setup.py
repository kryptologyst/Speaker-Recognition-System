"""
Setup script for Speaker Recognition System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="speaker-recognition-system",
    version="1.0.0",
    author="AI Developer",
    author_email="developer@example.com",
    description="Advanced deep learning system for speaker identification and verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/speaker-recognition-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchaudio[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "speaker-recognition=ui.streamlit_app:main",
            "train-speaker-model=training.trainer:main",
            "setup-database=data.database:create_sample_database",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
