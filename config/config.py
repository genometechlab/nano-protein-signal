"""
Configuration file for nanopore protein signal analysis
Modify parameters here for different analysis needs
"""

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
MIN_LENGTH = None # 1751 (05th percentile) Set to None to disable
MAX_LENGTH = None  # 4570 (95th percentile) Set to None to disable

# Parallel processing
N_JOBS = 18

# Visualization
COLOR_CYCLE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
FIG_SIZE_FULL = (20, 24)
FIG_SIZE_SINGLE = (12, 6)