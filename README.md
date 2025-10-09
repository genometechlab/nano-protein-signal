Computational framework for analyzing nanopore ionic current signals from protein translocation 

## Overview

This repository provides a computational framework for analyzing ionic current traces from protein constructs translocated through nanopore using the ClpX motor. Our pipeline segments these signals to identify distinct regions and classifies amino acid regions based on their ionic current signatures.


## Repository Structure
├── preprocessing/	# Denoising raw ionic current signal
├── segmentation/          # Change point detection and signal segmentation
├── features/              # Statistical feature extraction from segments
├── dtw/                   # Dynamic Time Warping visualization, classification and barycenter averaging
├── lstm/                  # Deep learning classification using LSTM
├── hmm/                   # Hidden Markov Model-based sequence analysis
└── visualization/         # Signal plotting and alignment visualization tools

## Analysis Pipeline

1. **Preprocessing**: Denoising raw ionoic current signal
2. **Segmentation**: Detect boundaries in ionic current traces using PELT and Dynamic Programming algorithms with custom variance-based cost function
3. **Feature Extraction**: Compute statistical properties (mean, std, skewness, kurtosis, etc.) for each segment
3. **DTW Alignment**: Constructing Barycenter for different classes of amino acids and DTW alignment
4. **Classification**: Identify amino acids regions using LSTM sequence classification
5. **Modeling**: Apply HMMs to capture temporal dependencies in translocation signals

## Requirements

- Python 3.8+
- ruptures, tslearn, PyTorch, scipy, numpy, scikit-learn

## Citation

## Licence 
