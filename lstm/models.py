"""
PyTorch Dataset for segmented trace features
"""

import torch
from torch.utils.data import Dataset


class SegmentTraceDataset(Dataset):
    """
    Dataset for segment-level features from nanopore traces
    
    Parameters:
    -----------
    traces : Tensor
        Padded feature tensors (batch_size, max_length, n_features)
    labels : Tensor
        Class labels
    lengths : Tensor
        Actual sequence lengths before padding
    """
    
    def __init__(self, traces, labels, lengths):
        self.traces = traces
        self.labels = labels
        self.lengths = lengths
    
    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx], self.lengths[idx]
    
    def __len__(self):
        return len(self.labels)