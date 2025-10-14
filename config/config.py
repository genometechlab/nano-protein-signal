"""
Configuration file for nanopore protein signal analysis
Modify parameters here for different analysis needs
"""
#####################################Segmentation###########################################
# Data paths
DATA_PATH = "./data/raw_denoised_all_PASTOR_boundries.json"
OUTPUT_DIR = "./output"

# PASTOR sequences
PASTORS = ["HDKER", "GNQST", "FYWCP", "AVLIM", "VGDNY", "TWAFH", "PRMQE", "KSILC"]

# Valid amino acids
VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

# Amino acid to class mapping
AA_CLASS_MAP = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
    'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

# Bessel filter parameters
FILTER_ORDER = 1
CUTOFF_FREQUENCY = 1500
SAMPLING_RATE = 3012

# Segmentation parameters - PELT
PELT_PENALTY = 5
PELT_MIN_SIZE = 30
PELT_SCALE = 1

# Segmentation parameters - Dynamic Programming
DYNP_N_BKPS = 34
DYNP_MIN_SIZE = 15
DYNP_SCALE = 1

# Length filtering (set to None to disable)
MIN_LENGTH = 1751 # 1751 (05th percentile) Set to None to disable
MAX_LENGTH = 4570  # 4570 (95th percentile) Set to None to disable

# Parallel processing
N_JOBS = 18

# Visualization
COLOR_CYCLE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
FIG_SIZE_FULL = (20, 24)
FIG_SIZE_SINGLE = (12, 6)

#####################################DTW###########################################
# DTW/DBA parameters
FIXED_SEG_LEN = 80  # Length to interpolate segments for DBA
FIXED_YLIM = (-4, 4)  # Y-axis limits for centroid plots

# Grid search parameters
COARSE_GRID = [(a, 1.0 - a) for a in np.linspace(0.5, 1.0, 6)]
FINE_GRID = [(a, 1.0 - a) for a in np.linspace(0.7, 1.0, 11)]

# DBA trace filtering (set to None to disable)
DBA_MIN_SEGMENTS = None  # Minimum number of segments per trace
DBA_MAX_SEGMENTS = None  # Maximum number of segments per trace
DBA_MIN_TRACE_LENGTH = None  # Minimum total trace length
DBA_MAX_TRACE_LENGTH = None  # Maximum total trace length

# Amino acids for DBA (set to None for all available)
DBA_TARGET_AAS = None  # e.g. ['A', 'D'] or None for all

# Cross-validation
N_FOLDS = 3

# Output directory for DTW results
DTW_OUTPUT_DIR = "./output/dtw"

#####################################LSTM###########################################
# LSTM parameters
LSTM_INPUT_SIZE = 8  # Number of features per segment
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True

# Training parameters
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_GRAD_CLIP = 1.0

# Cross-validation
LSTM_N_FOLDS = 5
LSTM_RANDOM_SEED = 42

# Pairwise classification
PAIRWISE_EPOCHS = 50
PAIRWISE_TEST_SIZE = 0.2

# Amino acid groupings
AA_GROUPS = {
    # Charge-based
    'positive': [7, 8, 14],  # H, K, R
    'negative': [2, 3],      # D, E
    'polar': [15, 16, 11, 13],  # S, T, N, Q
    'nonpolar': [0, 17, 9, 7, 12],  # A, V, L, I, P
    'aromatic': [4, 18, 19],  # F, W, Y
    'charged': [7, 8, 14, 2, 3],  # H, K, R, D, E
    
    # Size-based (by molecular weight and volume)
    'very_small': [5, 0, 15],  # G, A, S (< 90 Da)
    'small': [1, 2, 11, 12, 16, 17],  # C, D, N, P, T, V (90-120 Da)
    'medium': [3, 7, 9, 13],  # E, I, L, Q (120-140 Da)
    'large': [6, 8, 10, 4],  # H, K, M, F (140-165 Da)
    'very_large': [14, 18, 19]  # R, W, Y (> 165 Da)
}

# Amino acid properties (for reference)
AA_PROPERTIES = {
    'G': {'mw': 75.07, 'size': 'very_small', 'polarity': 'nonpolar'},
    'A': {'mw': 89.09, 'size': 'very_small', 'polarity': 'nonpolar'},
    'S': {'mw': 105.09, 'size': 'very_small', 'polarity': 'polar'},
    'C': {'mw': 121.16, 'size': 'small', 'polarity': 'polar'},
    'D': {'mw': 133.10, 'size': 'small', 'polarity': 'charged'},
    'N': {'mw': 132.12, 'size': 'small', 'polarity': 'polar'},
    'P': {'mw': 115.13, 'size': 'small', 'polarity': 'nonpolar'},
    'T': {'mw': 119.12, 'size': 'small', 'polarity': 'polar'},
    'V': {'mw': 117.15, 'size': 'small', 'polarity': 'nonpolar'},
    'E': {'mw': 147.13, 'size': 'medium', 'polarity': 'charged'},
    'I': {'mw': 131.17, 'size': 'medium', 'polarity': 'nonpolar'},
    'L': {'mw': 131.17, 'size': 'medium', 'polarity': 'nonpolar'},
    'Q': {'mw': 146.15, 'size': 'medium', 'polarity': 'polar'},
    'H': {'mw': 155.15, 'size': 'large', 'polarity': 'charged'},
    'K': {'mw': 146.19, 'size': 'large', 'polarity': 'charged'},
    'M': {'mw': 149.21, 'size': 'large', 'polarity': 'nonpolar'},
    'F': {'mw': 165.19, 'size': 'large', 'polarity': 'aromatic'},
    'R': {'mw': 174.20, 'size': 'very_large', 'polarity': 'charged'},
    'W': {'mw': 204.23, 'size': 'very_large', 'polarity': 'aromatic'},
    'Y': {'mw': 181.19, 'size': 'very_large', 'polarity': 'aromatic'}
}

# Class labels
AA_CLASS_MAP = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
    'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19
}

IDX_TO_AA = {v: k for k, v in AA_CLASS_MAP.items()}

# Output directory for LSTM results
LSTM_OUTPUT_DIR = "./output/lstm"