from setuptools import setup, find_packages

setup(
    name="vrhmm",
    version="1.0.0",
    description="Variable Rate Hidden Markov Model for Nanopore Sequencing",
    packages=["vrhmm", "vrhmm.cli", "vrhmm.config", "vrhmm.core", "vrhmm.io", 
              "vrhmm.processing", "vrhmm.segmentation", "vrhmm.utils", 
              "vrhmm.visualization", "vrhmm.yahmm"],
    package_dir={"vrhmm": "."},
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
    entry_points={
        "console_scripts": [
            "vrhmm=vrhmm.cli.main:main",
        ],
    },
)
