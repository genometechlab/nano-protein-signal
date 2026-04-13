from setuptools import setup, find_packages
from pathlib import Path

long_description = ""
readme = Path("README.md")
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

setup(
    name="vrhmm",
    version="1.0.0",
    description="Variable Region Hidden Markov Model for Nanopore Sequencing of Peptides",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",           
    url="",              
    license="",          
    packages=find_packages(),
    python_requires=">=3.10",
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
        "dtw": ["dtaidistance>=2.3.0"],
        "network": ["networkx>=2.6.0"],
    },
    entry_points={
        "console_scripts": [
            "vrhmm=vrhmm.cli.main:main",
        ],
    },
)