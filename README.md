Computational framework for analyzing nanopore ionic current signals from protein translocation 

## Overview

This repository provides a computational framework for analyzing ionic current traces from protein constructs translocated through nanopore using the ClpX motor. Our pipeline segments these signals to identify distinct regions and classifies amino acid regions based on their ionic current signatures.


## Repository Structure
```

nano_protein_signal/
├── config/	        # All parameters
│   └── config.py
├── preprocessing/      # Denoising raw ionic current signal
│   ├── 
├── segmentation/       # Change point detection and signal segmentation
│   ├── filters.py
│   ├── cost_functions.py
│   ├── segment_pelt.py
│   └── segment_dynp.py
├── features/       # Statistical feature extraction from segments
│   └── extract_features.py
├── dtw/        # Dynamic Time Warping visualization, classification and barycenter averaging
│   ├── preprocessing
│   ├── barycenter
│   └── classification
├── lstm/       # Deep learning classification using LSTM
│   ├── models.py
│   ├── dataset.py
│   ├── train.py
│   ├── run_multiclass.py       # 20-way amino acid classification
│   ├── run_pairwise.py         # All pairwise combinations
│   └── run_multigroup.py       # N-way group classification
├── hmm/        # Hidden Markov Model-based sequence analysis
│   ├── 
│   ├── 
├── visualization/      # Signal plotting and alignment visualization tools
│   ├── plot_full_pastor.py
│   ├── plot_full_pastor.py
│   ├── plot_dba_centroids.py
│   ├── plot_dtw_alignment.py
│   └── 
├── utils/
│   ├── data_loader.py
│   └── dtw_utils
├── requirements        # required libraries
└── README.md


```

## Analysis Pipeline

1. **Preprocessing**: Denoising raw ionoic current signal
2. **Segmentation**: Detect boundaries in ionic current traces using PELT and Dynamic Programming algorithms with custom variance-based cost function
3. **Feature Extraction**: Compute statistical properties (mean, std, skewness, kurtosis, etc.) for each segment
3. **DTW Alignment**: Constructing Barycenter for different classes of amino acids and DTW alignment
4. **Classification**: Identify amino acids regions using LSTM sequence classification
5. **Modeling**: Apply HMMs to capture temporal dependencies in translocation signals



## Citation

## Licence 
