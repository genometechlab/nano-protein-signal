"""Pipeline runner for vrHMM processing."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Correct imports based on your refactored structure
from vrhmm.core.hmm_builder import HMMConstructor
from vrhmm.core.classifier import HMMClassifier
from vrhmm.segmentation.segmenter import Segmenter, SegmentVarianceCollector
from vrhmm.io.loader import DataLoader
from vrhmm.io.writer import ResultWriter
from vrhmm.processing.signal_processor import SignalProcessor
from vrhmm.utils.amino_acids import (
    get_all_categories,
    get_amino_acids_in_category,
    get_amino_acid_category
)

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Manages the complete vrHMM processing pipeline."""

    def __init__(self, args: Any, config: Dict[str, Any]) -> None:
        
        self.args = args
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segmenter = Segmenter(config)
        self.processor = SignalProcessor(config)
        self.writer = ResultWriter(self.output_dir, self.timestamp)

        # Check if this is a testing run (model_aa and test_aa are both provided)
        self.is_testing_mode = (
                hasattr(args, 'test_aa') and
                hasattr(args, 'model_aa') and
                args.test_aa is not None and
                args.model_aa is not None
        )

        # Cross-validation is when they're different
        self.is_cross_validation = (
                self.is_testing_mode and
                args.test_aa != args.model_aa
        )

    def run(self) -> None:
        """Execute the complete pipeline."""
        logger.info("=" * 70)
        logger.info("vrHMM Classification Pipeline")
        logger.info(f"Classification Mode: {self.args.classification_mode}")
        logger.info(f"Variance Mode: {self.args.variance_mode}")

        if self.is_cross_validation:
            logger.info(f"CROSS-VALIDATION MODE: Testing {self.args.test_aa} with {self.args.model_aa} model")
        elif self.is_testing_mode:
            logger.info(f"TESTING MODE: {self.args.test_aa} data with {self.args.model_aa} model")

        logger.info("=" * 70)

        # Load data
        barycenters = self._load_barycenters()
        signal_data = self._load_signals()

        if not signal_data:
            logger.error("No signal data loaded")
            return

        # Collect variances if needed
        variance_collectors = None
        if self.args.variance_mode in ['segment', 'empirical']:
            variance_collectors = self._collect_variances(signal_data, list(barycenters.keys()))

        # Build classifier
        classifier = self._build_classifier(barycenters, variance_collectors)

        # Process signals
        results, df = self._process_signals(signal_data, classifier)

        # Save results
        self.writer.save_results(results, df, self.args)

        # Generate visualizations
        if not self.args.no_plots:
            self._generate_visualizations(results, df, barycenters, variance_collectors)

        logger.info("=" * 70)
        logger.info("Pipeline Complete")
        logger.info(f"Results saved to: {self.output_dir}")

        # Print accuracy
        if 'predicted_category' in df.columns:
            self._print_test_results(df)

        logger.info("=" * 70)

    def _load_barycenters(self) -> Dict[str, List[np.ndarray]]:
        
        if self.args.barycenter_file:
            loader = DataLoader(str(self.args.barycenter_file), 'json')
            data = loader.load_data()
            if data:
                valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
                return {k: v for k, v in data.items() if k in valid_aas}
        elif self.args.data_dir:
            return self._load_barycenters_from_dir()
        else:
            raise ValueError("Must provide --barycenter-file or --data-dir")
        return {}

    def _load_barycenters_from_dir(self) -> Dict[str, List[np.ndarray]]:
        
        barycenters = {}
        pattern = self.args.data_dir / "dba_{aa}_barycenter.json"

        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            path = Path(str(pattern).format(aa=aa.lower()))
            if path.exists():
                loader = DataLoader(str(path), 'json')
                data = loader.load_data()
                if data and aa in data:
                    barycenters[aa] = data[aa]

        return barycenters

    def _load_signals(self) -> List[Dict[str, Any]]:
        
        if self.args.signal_file:
            signal_path = Path(self.args.signal_file)
        elif self.args.data_dir:
            signal_path = Path(self.args.data_dir) / "signals.csv"
        else:
            raise ValueError("Must provide --signal-file or --data-dir")

        if not signal_path.exists():
            logger.error(f"Signal file not found: {signal_path}")
            return []

        file_extension = signal_path.suffix.lower()
        is_pickle = file_extension in ['.pkl', '.pickle']
        data_type = 'pickle' if is_pickle or self.args.use_pickle else 'csv'

        # Load metadata FIRST, before creating DataLoader
        metadata = None
        if hasattr(self.args, 'metadata_file') and self.args.metadata_file:
            metadata_path = Path(self.args.metadata_file)
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path.name}")
                if 'traces' in metadata:
                    logger.info(f"Metadata specifies {len(metadata['traces'])} traces to analyze")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")

        # NOW create the DataLoader with the metadata
        loader = DataLoader(
            str(signal_path),
            data_type,
            signal_dict=True,
            metadata=metadata,  # Pass it here!
            min_signal_length=self.args.min_signal_length,
            max_signal_length=self.args.max_signal_length
        )

        data = loader.load_data()

        if data:
            logger.info(f"Loaded {len(data)} signals from {signal_path.name}")

            # Filter for test amino acid if specified (this happens AFTER metadata filtering now)
            if hasattr(self.args, 'test_aa') and self.args.test_aa:
                original_count = len(data)
                data = [s for s in data if s.get('aa') == self.args.test_aa]
                logger.info(f"Filtered to {len(data)} signals for amino acid {self.args.test_aa}")

        return data or []

    def _collect_variances(
            self,
            signal_data: List[Dict[str, Any]],
            amino_acids: List[str]
    ) -> Dict[str, SegmentVarianceCollector]:
        """Collect segment variances from signals."""
        collectors = {aa: SegmentVarianceCollector() for aa in amino_acids}

        aa_signals = defaultdict(list)
        for record in signal_data:
            aa = record.get('aa', '')
            if aa in amino_acids:
                aa_signals[aa].append(record)

        logger.info("Collecting segment variances from actual signals:")

        for aa in amino_acids:
            signals = aa_signals[aa][:10]  # Use first 10 signals

            if not signals:
                continue

            processed = 0
            for record in signals:
                try:
                    # Check if pre-segmented
                    raw_data = record['cleaned_segment']
                    if isinstance(raw_data, list) and len(raw_data) > 0:
                        if isinstance(raw_data[0], (list, np.ndarray)):
                            # Pre-segmented data
                            variances = []
                            for seg in raw_data:
                                if seg is not None:
                                    seg_array = np.array(seg).flatten()
                                    if len(seg_array) > 0:
                                        variances.append(float(np.var(seg_array)))
                            if len(variances) == 35:
                                collectors[aa].add_signal_variances(variances)
                                processed += 1
                        else:
                            # Raw signal
                            signal = self.processor.parse_signal(raw_data)
                            result = self.segmenter.segment(signal, self.args.seg_mode)
                            collectors[aa].add_signal_variances(result['variances'].tolist())
                            processed += 1
                except Exception as e:
                    logger.debug(f"Error processing signal for variance: {e}")

            if processed > 0:
                logger.info(f"  {aa}: Collected variances from {processed} signals")

        return collectors

    def _build_classifier(
            self,
            barycenters: Dict[str, List[np.ndarray]],
            variance_collectors: Optional[Dict[str, SegmentVarianceCollector]]
    ) -> HMMClassifier:
        """Build HMM classifier models."""
        classifier = HMMClassifier(self.args.classification_mode)

        hmm_config = self.config['hmm'].copy()
        if self.args.variance_mode == 'barycenter':
            hmm_config['variance_scale'] = self.args.variance_scale
            constructor = HMMConstructor(hmm_config, variance_mode='barycenter',
                                         variance_scale=self.args.variance_scale)
        else:
            constructor = HMMConstructor(hmm_config, variance_mode='segment', variance_scale=1.0)

        logger.info(f"Building models for amino acids: {sorted(barycenters.keys())}")

        for aa in barycenters:
            profile_arrays = barycenters[aa]

            segment_variances = None
            if self.args.variance_mode in ['segment',
                                           'empirical'] and variance_collectors and aa in variance_collectors:
                segment_variances = variance_collectors[aa].get_average_variances()
                logger.debug(f"Using empirical variances for {aa}")

            try:
                model = constructor.build_hmm_from_arrays(
                    aa, profile_arrays, segment_variances
                )
                classifier.add_model(aa, model)
                logger.debug(f"Built model for {aa}")
            except Exception as e:
                logger.error(f"Error building model for {aa}: {e}")

        logger.info(f"Built {len(classifier.hmm_models)} HMM models")
        return classifier

    def _process_signals(
            self,
            signal_data: List[Dict[str, Any]],
            classifier: HMMClassifier
    ) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
        """Process and classify signals."""
        if self.args.testing_mode:
            signal_data = signal_data[:self.args.test_limit]
            logger.info(f"Testing mode: Processing only {len(signal_data)} signals")

        results = {}
        for i, record in enumerate(signal_data, 1):
            try:
                result = self.processor.process_signal(
                    record,
                    self.segmenter,
                    classifier,
                    self.args.seg_mode
                )

                signal_key = f"{record['run']}_{record['channel']}_{record.get('segment', i)}_{record.get('aa', 'unknown')}"
                results[signal_key] = result

                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(signal_data)} signals")

            except Exception as e:
                logger.debug(f"Error processing signal {i}: {e}")

        logger.info(f"Successfully processed {len(results)} signals")

        df = self._create_summary_dataframe(results)
        return results, df

    def _create_summary_dataframe(
            self,
            results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        
        rows = []
        for signal_key, result in results.items():
            parts = signal_key.split('_')
            rows.append({
                'signal_key': signal_key,
                'run': parts[0] if len(parts) > 0 else 'unknown',
                'channel': parts[1] if len(parts) > 1 else 'unknown',
                'segment': parts[2] if len(parts) > 2 else 'unknown',
                'true_aa': result['amino_acid'],
                'predicted_category': result.get('predicted_category', ''),
                'log_probability': result.get('log_probability', 0.0),
                'num_segments': result['num_segments'],
                'signal_length': result['signal_length']
            })

        return pd.DataFrame(rows)

    def _generate_visualizations(
            self,
            results: Dict[str, Dict[str, Any]],
            df: pd.DataFrame,
            barycenters: Dict[str, List[np.ndarray]],
            variance_collectors: Optional[Dict[str, SegmentVarianceCollector]]
    ) -> None:
        """Generate visualization plots."""
        plot_dir = self.output_dir / "visualizations"
        plot_dir.mkdir(exist_ok=True)

        # Generate classification plots ONLY if NOT in testing mode
        if not self.is_testing_mode and 'predicted_category' in df.columns:
            try:
                from vrhmm.visualization import generate_classification_report
                generate_classification_report(
                    df,
                    self.args.classification_mode,
                    str(plot_dir)
                )
                logger.info(f"Saved classification visualizations to {plot_dir}")
            except ImportError:
                logger.warning("Classification visualization modules not available")
        elif self.is_testing_mode:
            logger.info("Testing mode: Skipping classification visualizations")

        # Generate signal/HMM plots
        try:
            from vrhmm.visualization.signal_plots import (
                plot_hmm_segmentation_and_path,
                plot_multi_panel_hmm_states,
                plot_segment_pileup,
                plot_match_state_pileup,
                plot_backslip_distribution,
                plot_skip_distribution,
                plot_backslip_by_position
            )

            logger.info("Generating HMM signal visualizations...")

            if results:
                # Check if metadata filtering was used
                using_metadata = hasattr(self.args, 'metadata_file') and self.args.metadata_file is not None

                if using_metadata:
                    # Plot EVERY trace individually when using metadata
                    logger.info(
                        f"Metadata filtering detected - generating individual plots for all {len(results)} traces")
                    individual_plots_dir = plot_dir / "individual_traces"
                    individual_plots_dir.mkdir(exist_ok=True)

                    # Sort by signal key for consistent ordering
                    sorted_keys = sorted(results.keys())

                    for idx, signal_key in enumerate(sorted_keys, 1):
                        result = results[signal_key]
                        amino_acid = result.get('amino_acid', 'unknown')
                        log_prob = result.get('log_probability', float('-inf'))

                        # Extract run and channel for filename
                        parts = signal_key.split('_')
                        run = parts[0] if len(parts) > 0 else 'unknown'
                        channel = parts[1] if len(parts) > 1 else 'unknown'

                        # Create descriptive filename
                        trace_filename = f"{idx:02d}_{run}_Ch{channel}_{amino_acid}_LogP{log_prob:.1f}.pdf"
                        trace_path = individual_plots_dir / trace_filename

                        plot_hmm_segmentation_and_path(
                            signal=result['z_normalized_signal'],
                            segment_results=result['segment_results'],
                            state_sequence=result.get('state_sequence', []),
                            full_path=result.get('full_path', []),
                            signal_key=signal_key,
                            save_path=str(trace_path)
                        )

                        if idx % 5 == 0:
                            logger.info(f"  Generated {idx}/{len(results)} individual plots")

                    logger.info(f"✓ Saved {len(results)} individual trace plots to {individual_plots_dir}")

                else:
                    # Original behavior - just plot the best trace
                    best_signal_key = max(
                        results.keys(),
                        key=lambda k: results[k].get('log_probability', float('-inf'))
                    )
                    best_result = results[best_signal_key]

                    # Plot best trace
                    best_trace_path = plot_dir / f"best_trace_{self.args.test_aa}_{self.timestamp}.pdf"
                    plot_hmm_segmentation_and_path(
                        signal=best_result['z_normalized_signal'],
                        segment_results=best_result['segment_results'],
                        state_sequence=best_result.get('state_sequence', []),
                        full_path=best_result.get('full_path', []),
                        signal_key=f"BEST - {best_signal_key}",
                        save_path=str(best_trace_path)
                    )
                    logger.info(f"Saved best trace plot: {best_trace_path.name}")

                # Multi-panel comparison (always generate)
                multi_panel_path = plot_dir / f"hmm_multi_panel_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_multi_panel_hmm_states(
                    results,
                    max_panels=min(10, len(results)) if using_metadata else 10,
                    save_path=str(multi_panel_path),
                    title=f"HMM State Alignment - {self.args.test_aa}"
                )

                # Segment pileup
                pileup_path = plot_dir / f"segment_pileup_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_segment_pileup(
                    results,
                    max_traces=min(10, len(results)),
                    save_path=str(pileup_path),
                    amino_acid=self.args.test_aa
                )

                # Prepare profile stats for match state pileup
                profile_stats = None
                if self.args.model_aa in barycenters:
                    profile_arrays = barycenters[self.args.model_aa]
                    profile_stats = {}

                    if self.args.variance_mode in ['segment', 'empirical'] and variance_collectors:
                        if self.args.model_aa in variance_collectors:
                            empirical_variances = variance_collectors[self.args.model_aa].get_average_variances()
                            for i in range(min(len(profile_arrays), len(empirical_variances))):
                                seg_mean = float(np.mean(profile_arrays[i]))
                                emp_var = empirical_variances[i]
                                if isinstance(emp_var, (list, np.ndarray)):
                                    emp_var = float(np.mean(emp_var))
                                else:
                                    emp_var = float(emp_var)
                                seg_std = float(np.sqrt(emp_var))
                                profile_stats[str(i)] = (seg_mean, seg_std)
                    else:
                        # Use barycenter variances
                        for i, seg_array in enumerate(profile_arrays):
                            seg_mean = float(np.mean(seg_array))
                            seg_var = float(np.var(seg_array))
                            if self.args.variance_mode == 'barycenter':
                                seg_var = seg_var * self.args.variance_scale
                            seg_std = float(np.sqrt(seg_var))
                            profile_stats[str(i)] = (seg_mean, seg_std)

                # Match state pileup
                match_pileup_path = plot_dir / f"match_state_pileup_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_match_state_pileup(
                    results,
                    amino_acid=self.args.test_aa,
                    barycenter_profile_stats=profile_stats,
                    save_path=str(match_pileup_path)
                )

                # State distribution plots
                backslip_path = plot_dir / f"backslip_distribution_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_backslip_distribution(
                    results,
                    amino_acid=self.args.test_aa,
                    save_path=str(backslip_path),
                    title=f"Backslip Distribution - {self.args.test_aa}"
                )

                skip_path = plot_dir / f"skip_distribution_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_skip_distribution(
                    results,
                    amino_acid=self.args.test_aa,
                    save_path=str(skip_path),
                    title=f"Skip Distribution - {self.args.test_aa}"
                )

                # Backslip by position
                backslip_pos_path = plot_dir / f"backslip_by_position_{self.args.test_aa}_{self.timestamp}.pdf"
                plot_backslip_by_position(
                    results,
                    amino_acid=self.args.test_aa,
                    save_path=str(backslip_pos_path),
                    title=f"Backslip/Skip Position Analysis - {self.args.test_aa}"
                )

                logger.info("HMM signal visualization generation complete")

        except ImportError as e:
            logger.warning(f"Signal visualization modules not available: {e}")

    def _print_test_results(self, df: pd.DataFrame) -> None:
        """Print test results."""
        if self.args.classification_mode != '20way':
            df['true_category'] = df['true_aa'].apply(
                lambda aa: get_amino_acid_category(aa, self.args.classification_mode)
            )
        else:
            df['true_category'] = df['true_aa']

        correct = (df['true_category'] == df['predicted_category']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        logger.info("")
        if self.is_cross_validation:
            logger.info("CROSS-VALIDATION RESULTS")
        else:
            logger.info("TEST RESULTS")

        logger.info(f"Model AA: {self.args.model_aa}")
        logger.info(f"Test AA: {self.args.test_aa}")
        logger.info(f"Total Signals: {total}")
        logger.info(f"Correct Predictions: {correct}")
        logger.info(f"Accuracy: {accuracy:.2%}")