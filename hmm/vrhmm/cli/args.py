"""Argument parser for the vrhmm CLI."""

import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description='vrhmm: Variable Rate HMM for Amino Acid Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_classification_args(parser)
    add_model_args(parser)
    add_data_args(parser)
    add_output_args(parser)
    add_processing_args(parser)
    add_advanced_args(parser)

    return parser


def add_classification_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Classification Configuration')

    group.add_argument(
        '--classification-mode',
        type=str,
        default='20way',
        choices=['2way', '3way', '4way', '5way_size', 'biological', '20way'],
        help='Classification mode'
    )

    group.add_argument(
        '--variance-mode',
        type=str,
        default='barycenter',
        choices=['barycenter', 'segment'],
        help='Variance calculation mode'
    )

    group.add_argument(
        '--variance-scale',
        type=float,
        default=1.0,
        help='Variance scaling factor'
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Model Configuration')

    group.add_argument(
        '--model-aa',
        type=str,
        default=None,
        help='Reference amino acid model'
    )

    group.add_argument(
        '--test-aa',
        type=str,
        default=None,
        help='Amino acid data to test'
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Data Sources')

    group.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Base directory for data files'
    )

    group.add_argument(
        '--signal-file',
        type=Path,
        default=None,
        help='Path to signal data file'
    )

    group.add_argument(
        '--metadata-file',
        type=Path,
        default=None,
        help='Path to metadata JSON file specifying which traces to analyze'
    )

    group.add_argument(
        '--barycenter-file',
        type=Path,
        default=None,
        help='Path to barycenter JSON file'
    )

    group.add_argument(
        '--profile-file',
        type=Path,
        default=None,
        help='Path to pre-computed profile CSV with columns: amino_acid, state, mean, std'
    )

    group.add_argument(
        '--use-pickle',
        action='store_true',
        help='Use pre-segmented pickle data'
    )

    group.add_argument(
        '--variance-scale-file',
        type=Path,
        default=None,
        help='CSV file with per-AA variance scales (columns: amino_acid, variance_scale)'
    )

    group.add_argument(
        '--transition-file',
        type=str,
        default=None,
        help='Path to JSON file with custom transition probabilities'
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Output Configuration')

    group.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results'),
        help='Output directory'
    )

    group.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    group.add_argument(
        '--save-reorganized',
        action='store_true',
        help='Save HMM-reorganized segments'
    )


def add_processing_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Signal Processing')

    group.add_argument(
        '--seg-mode',
        type=str,
        default='dynp',
        choices=['dynp', 'set_window', 'pelt'],
        help='Segmentation mode'
    )

    group.add_argument(
        '--min-signal-length',
        type=int,
        default=1751,
        help='Minimum signal length'
    )

    group.add_argument(
        '--max-signal-length',
        type=int,
        default=4570,
        help='Maximum signal length'
    )


def add_advanced_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Advanced Options')

    group.add_argument(
        '--testing-mode',
        action='store_true',
        help='Enable testing mode'
    )

    group.add_argument(
        '--test-limit',
        type=int,
        default=5,
        help='Number of signals in testing mode'
    )

    group.add_argument(
        '--backslip-mode',
        type=str,
        default='ignore',
        choices=['ignore', 'delete', 'average'],
        help='Backslip handling mode'
    )

    group.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Custom configuration file'
    )