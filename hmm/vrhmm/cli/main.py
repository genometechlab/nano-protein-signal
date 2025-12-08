"""Main entry point for vrhmm CLI."""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from vrhmm.cli.args import create_parser
from vrhmm.cli.runner import PipelineRunner
from vrhmm.config import CONFIG
from vrhmm.utils.types import ClassificationResult

def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for vrhmm pipeline."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    setup_logging(parsed_args.testing_mode)
    logger = logging.getLogger(__name__)

    try:
        runner = PipelineRunner(parsed_args, CONFIG)
        runner.run()
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())