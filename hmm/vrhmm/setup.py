"""Setup configuration for vrhmm package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vrhmm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Variable Rate Hidden Markov Model for Nanopore RNA Sequencing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vrHMM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "ruptures>=1.1.0",
        "numba>=0.54.0",
        "orjson>=3.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
        ],
        "dtw": ["dtaidistance>=2.3.0"],
    },
    entry_points={
        "console_scripts": [
            "vrhmm=vrhmm.cli.main:main",
        ],
    },
)