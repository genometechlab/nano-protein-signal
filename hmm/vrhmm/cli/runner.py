"""Pipeline runner for vrHMM processing."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

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

STANDARD_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


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

        self.is_testing_mode = (
            args.test_aa is not None and
            args.model_aa is not None
        )

        self.is_cross_validation = (
            self.is_testing_mode and
            args.test_aa != args.model_aa
        )

        self._load_transitions()

    def _load_transitions(self) -> None:
        """Load custom transition probabilities from file into config."""
        if not self.args.transition_file:
            return

        trans_path = Path(self.args.transition_file)
        if not trans_path.exists():
            logger.warning(f"Transition file not found: {trans_path}")
            return

        with open(trans_path, 'r') as f:
            transitions = json.load(f)

        self.config.setdefault('hmm', {}).setdefault('transitions', {})
        self.config['hmm']['transitions'].update(transitions)

        logger.info(f"Loaded custom transitions from {trans_path.name}")
        for key, value in sorted(transitions.items()):
            logger.info(f"  {key}: {value:.6f}")

    def _load_variance_scales(self) -> Dict[str, float]:
        default_scale = self.args.variance_scale
        scales = {aa: default_scale for aa in STANDARD_AMINO_ACIDS}

        if self.args.variance_scale_file:
            scale_path = Path(self.args.variance_scale_file)
            if scale_path.exists():
                df = pd.read_csv(scale_path)
                for _, row in df.iterrows():
                    scales[row['amino_acid']] = float(row['variance_scale'])
                logger.info(f"Loaded per-AA variance scales from {scale_path.name}")
            else:
                logger.warning(f"Variance scale file not found: {scale_path}")

        return scales

    def run(self) -> None:
        logger.info("vrHMM Classification Pipeline")
        logger.info(f"Classification Mode: {self.args.classification_mode}")
        logger.info(f"Variance Mode: {self.args.variance_mode}")

        if self.is_cross_validation:
            logger.info(f"CROSS-VALIDATION MODE: Testing {self.args.test_aa} with {self.args.model_aa} model")
        elif self.is_testing_mode:
            logger.info(f"TESTING MODE: {self.args.test_aa} data with {self.args.model_aa} model")

        precomputed_profiles = self._load_profile_from_csv()
        barycenters = None

        if precomputed_profiles is not None:
            logger.info("Using pre-computed HMM profiles from CSV")
            classifier, all_profile_stats = self._build_classifier_from_profiles(precomputed_profiles)
        else:
            barycenters = self._load_barycenters()
            signal_data = self._load_signals()

            if not signal_data:
                logger.error("No signal data loaded")
                return

            variance_collectors = None
            if self.args.variance_mode in ('segment', 'empirical'):
                variance_collectors = self._collect_variances(signal_data, list(barycenters.keys()))

            classifier, all_profile_stats = self._build_classifier(barycenters, variance_collectors)
            self._save_profile_stats(all_profile_stats)

        signal_data = self._load_signals()
        if not signal_data:
            logger.error("No signal data loaded")
            return

        results, df = self._process_signals(signal_data, classifier)
        self.writer.save_results(results, df, self.args)

        if not self.args.no_plots:
            self._generate_visualizations(results, df, barycenters, None, classifier)

        logger.info("Pipeline Complete")
        logger.info(f"Results saved to: {self.output_dir}")

        if 'predicted_category' in df.columns:
            self._print_test_results(df)

    def _load_profile_from_csv(self) -> Optional[Dict[str, Dict[str, Tuple[float, float]]]]:
        if self.args.profile_file is None:
            return None

        profile_path = Path(self.args.profile_file)
        if not profile_path.exists():
            logger.error(f"Profile file not found: {profile_path}")
            return None

        logger.info(f"Loading profile from CSV: {profile_path}")
        df = pd.read_csv(profile_path)

        required_cols = {'amino_acid', 'state', 'mean', 'std'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"Profile CSV missing required columns: {missing}")
            return None

        all_profile_stats = {}
        for aa in df['amino_acid'].unique():
            aa_df = df[df['amino_acid'] == aa].sort_values('state')
            all_profile_stats[aa] = {
                str(row['state']): (float(row['mean']), float(row['std']))
                for _, row in aa_df.iterrows()
            }
            logger.info(f"  Loaded {aa}: {len(all_profile_stats[aa])} states")

        logger.info(f"Loaded profiles for {len(all_profile_stats)} amino acids from {profile_path.name}")
        return all_profile_stats

    def _build_classifier_from_profiles(
        self,
        all_profile_stats: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> Tuple[HMMClassifier, Dict[str, Dict[str, Tuple[float, float]]]]:
        classifier = HMMClassifier(self.args.classification_mode)
        hmm_config = self.config['hmm'].copy()
        variance_scales = self._load_variance_scales()
        default_scale = getattr(self.args, 'variance_scale', 1.0)

        if self.args.model_aa:
            aas_to_build = [self.args.model_aa]
            logger.info(f"Testing mode: Building model only for {self.args.model_aa}")
        else:
            aas_to_build = sorted(all_profile_stats.keys())

        logger.info(f"Building models from profile CSV: {aas_to_build}")

        for aa in aas_to_build:
            if aa not in all_profile_stats:
                logger.error(f"No profile stats found for {aa}")
                continue

            profile_stats = all_profile_stats[aa]
            aa_scale = variance_scales.get(aa, default_scale)

            constructor = HMMConstructor(
                hmm_config,
                variance_mode='barycenter',
                variance_scale=aa_scale
            )

            try:
                model, _ = constructor.build_hmm_from_profile_stats(aa, profile_stats)
                classifier.add_model(aa, model)
            except Exception as e:
                logger.error(f"Error building model for {aa}: {e}")

        logger.info(f"Built {len(classifier.hmm_models)} HMM models from profile CSV")
        return classifier, all_profile_stats

    def _save_profile_stats(
        self,
        all_profile_stats: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> None:
        rows = []
        for aa, profile_stats in sorted(all_profile_stats.items()):
            for i in range(len(profile_stats)):
                key = str(i)
                if key in profile_stats:
                    mean, std = profile_stats[key]
                    rows.append({
                        'amino_acid': aa,
                        'state': i,
                        'mean': mean,
                        'std': std,
                        'var': std ** 2
                    })

        csv_path = self.output_dir / f"hmm_profiles_{self.timestamp}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Saved HMM profile stats to: {csv_path}")

        json_output = {
            'timestamp': self.timestamp,
            'variance_mode': self.args.variance_mode,
            'profiles': {
                aa: {
                    'num_states': len(stats),
                    'states': {int(k): {'mean': v[0], 'std': v[1]} for k, v in stats.items()}
                }
                for aa, stats in all_profile_stats.items()
            }
        }

        json_path = self.output_dir / f"hmm_profiles_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)

    def _load_barycenters(self) -> Dict[str, List[np.ndarray]]:
        if self.args.barycenter_file:
            loader = DataLoader(str(self.args.barycenter_file), 'json')
            data = loader.load_data()
            if data:
                valid_aas = set(STANDARD_AMINO_ACIDS)
                return {k: v for k, v in data.items() if k in valid_aas}
        elif self.args.data_dir:
            return self._load_barycenters_from_dir()
        else:
            raise ValueError("Must provide --barycenter-file, --profile-file, or --data-dir")
        return {}

    def _load_barycenters_from_dir(self) -> Dict[str, List[np.ndarray]]:
        barycenters = {}
        pattern = self.args.data_dir / "dba_{aa}_barycenter.json"

        for aa in STANDARD_AMINO_ACIDS:
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

        is_pickle = signal_path.suffix.lower() in ('.pkl', '.pickle')
        data_type = 'pickle' if is_pickle or self.args.use_pickle else 'csv'

        metadata = self._load_metadata()

        loader = DataLoader(
            str(signal_path),
            data_type,
            signal_dict=True,
            metadata=metadata,
            min_signal_length=self.args.min_signal_length,
            max_signal_length=self.args.max_signal_length
        )

        data = loader.load_data()

        if data:
            logger.info(f"Loaded {len(data)} signals from {signal_path.name}")
            data = self._filter_signals(data)

        return data or []

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        if not self.args.metadata_file:
            return None

        metadata_path = Path(self.args.metadata_file)
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded metadata from {metadata_path.name}")
        if 'traces' in metadata:
            logger.info(f"Metadata specifies {len(metadata['traces'])} traces to analyze")

        return metadata

    def _filter_signals(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.args.test_aa:
            original_count = len(data)
            data = [s for s in data if s.get('aa') == self.args.test_aa]
            logger.info(f"Filtered to {len(data)}/{original_count} signals for amino acid {self.args.test_aa}")

        if self.args.classification_mode != '20way':
            valid_aas = set()
            for cat in get_all_categories(self.args.classification_mode):
                valid_aas.update(get_amino_acids_in_category(cat, self.args.classification_mode))
            original_count = len(data)
            data = [s for s in data if s.get('aa') in valid_aas]
            logger.info(f"Filtered to {len(data)}/{original_count} signals for {self.args.classification_mode} mode")

        return data

    def _collect_variances(
        self,
        signal_data: List[Dict[str, Any]],
        amino_acids: List[str]
    ) -> Dict[str, SegmentVarianceCollector]:
        collectors = {aa: SegmentVarianceCollector() for aa in amino_acids}

        aa_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in signal_data:
            aa = record.get('aa', '')
            if aa in amino_acids:
                aa_signals[aa].append(record)

        logger.info("Collecting segment variances from actual signals:")

        for aa in amino_acids:
            signals = aa_signals[aa][:10]
            if not signals:
                continue

            processed = 0
            for record in signals:
                try:
                    raw_data = record['cleaned_segment']
                    if isinstance(raw_data, list) and len(raw_data) > 0:
                        if isinstance(raw_data[0], (list, np.ndarray)):
                            variances = [
                                float(np.var(np.array(seg).flatten()))
                                for seg in raw_data
                                if seg is not None and len(np.array(seg).flatten()) > 0
                            ]
                            if variances:
                                collectors[aa].add_signal_variances(variances)
                                processed += 1
                        else:
                            signal = self.processor.parse_signal(raw_data)
                            result = self.segmenter.segment(signal, self.args.seg_mode)
                            collectors[aa].add_signal_variances(result['variances'].tolist())
                            processed += 1
                except Exception as e:
                    logger.warning(f"Error processing signal for variance ({aa}): {e}")

            if processed > 0:
                logger.info(f"  {aa}: Collected variances from {processed} signals")

        return collectors

    def _build_classifier(
        self,
        barycenters: Dict[str, List[np.ndarray]],
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]]
    ) -> Tuple[HMMClassifier, Dict[str, Dict[str, Tuple[float, float]]]]:
        classifier = HMMClassifier(self.args.classification_mode)
        all_profile_stats = {}
        variance_scales = self._load_variance_scales()
        hmm_config = self.config['hmm'].copy()

        logger.info(f"Building models for amino acids: {sorted(barycenters.keys())}")

        for aa, profile_arrays in barycenters.items():
            aa_scale = variance_scales.get(aa, self.args.variance_scale)

            constructor = HMMConstructor(
                hmm_config,
                variance_mode=self.args.variance_mode,
                variance_scale=aa_scale
            )

            segment_variances = None
            if self.args.variance_mode in ('segment', 'empirical') and variance_collectors and aa in variance_collectors:
                segment_variances = variance_collectors[aa].get_average_variances()

            try:
                model, profile_stats = constructor.build_hmm_from_arrays(
                    aa, profile_arrays, segment_variances
                )
                classifier.add_model(aa, model)
                all_profile_stats[aa] = profile_stats
            except Exception as e:
                logger.error(f"Error building model for {aa}: {e}")

        return classifier, all_profile_stats

    def _process_signals(
        self,
        signal_data: List[Dict[str, Any]],
        classifier: HMMClassifier
    ) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
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

                signal_key = (
                    f"{record['run']}_{record['channel']}_"
                    f"{record.get('segment', i)}_{record.get('aa', 'unknown')}"
                )
                results[signal_key] = result

                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(signal_data)} signals")

            except Exception as e:
                logger.warning(f"Error processing signal {i}: {e}")

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

    # Visualization
    def _generate_visualizations(
        self,
        results: Dict[str, Dict[str, Any]],
        df: pd.DataFrame,
        barycenters: Optional[Dict[str, List[np.ndarray]]],
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]],
        classifier: Optional[HMMClassifier] = None
    ) -> None:
        plot_dir = self.output_dir / "visualizations"
        plot_dir.mkdir(exist_ok=True)

        if not self.is_testing_mode and 'predicted_category' in df.columns:
            self._plot_classification_report(df, plot_dir)
            self._generate_pairwise_matrix(results, classifier, plot_dir)
        elif self.is_testing_mode:
            logger.info("Testing mode: Skipping classification visualizations")

        self._plot_hmm_diagnostics(results, barycenters, variance_collectors, plot_dir)

    def _plot_classification_report(self, df: pd.DataFrame, plot_dir: Path) -> None:
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

    def _plot_hmm_diagnostics(
        self,
        results: Dict[str, Dict[str, Any]],
        barycenters: Optional[Dict[str, List[np.ndarray]]],
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]],
        plot_dir: Path
    ) -> None:
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
        except ImportError as e:
            logger.warning(f"Signal visualization modules not available: {e}")
            return

        logger.info("Generating HMM signal visualizations...")

        if not results:
            return

        using_metadata = self.args.metadata_file is not None

        if using_metadata:
            self._plot_individual_traces(results, plot_dir, plot_hmm_segmentation_and_path)
        else:
            self._plot_best_trace(results, plot_dir, plot_hmm_segmentation_and_path)

        trace_count = min(10, len(results)) if using_metadata else 10

        multi_panel_path = plot_dir / f"hmm_multi_panel_{self.args.test_aa}_{self.timestamp}.pdf"
        plot_multi_panel_hmm_states(
            results,
            max_panels=trace_count,
            save_path=str(multi_panel_path),
            title=f"HMM State Alignment - {self.args.test_aa}"
        )

        pileup_path = plot_dir / f"segment_pileup_{self.args.test_aa}_{self.timestamp}.pdf"
        plot_segment_pileup(
            results,
            max_traces=min(10, len(results)),
            save_path=str(pileup_path),
            amino_acid=self.args.test_aa
        )

        profile_stats = self._compute_profile_stats_for_plot(barycenters, variance_collectors)

        match_pileup_path = plot_dir / f"match_state_pileup_{self.args.test_aa}_{self.timestamp}.pdf"
        plot_match_state_pileup(
            results,
            amino_acid=self.args.test_aa,
            barycenter_profile_stats=profile_stats,
            save_path=str(match_pileup_path)
        )

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

        backslip_pos_path = plot_dir / f"backslip_by_position_{self.args.test_aa}_{self.timestamp}.pdf"
        plot_backslip_by_position(
            results,
            amino_acid=self.args.test_aa,
            save_path=str(backslip_pos_path),
            title=f"Backslip/Skip Position Analysis - {self.args.test_aa}"
        )

        logger.info("HMM signal visualization generation complete")

    def _plot_individual_traces(self, results, plot_dir, plot_fn) -> None:
        logger.info(f"Generating individual plots for all {len(results)} traces")
        individual_dir = plot_dir / "individual_traces"
        individual_dir.mkdir(exist_ok=True)

        for idx, (signal_key, result) in enumerate(sorted(results.items()), 1):
            amino_acid = result.get('amino_acid', 'unknown')
            log_prob = result.get('log_probability', float('-inf'))

            parts = signal_key.split('_')
            run = parts[0] if parts else 'unknown'
            channel = parts[1] if len(parts) > 1 else 'unknown'

            filename = f"{idx:02d}_{run}_Ch{channel}_{amino_acid}_LogP{log_prob:.1f}.pdf"
            plot_fn(
                signal=result['z_normalized_signal'],
                segment_results=result['segment_results'],
                state_sequence=result.get('state_sequence', []),
                full_path=result.get('full_path', []),
                signal_key=signal_key,
                save_path=str(individual_dir / filename)
            )

            if idx % 5 == 0:
                logger.info(f"  Generated {idx}/{len(results)} individual plots")

        logger.info(f"Saved {len(results)} individual trace plots to {individual_dir}")

    def _plot_best_trace(self, results, plot_dir, plot_fn) -> None:
        best_key = max(
            results,
            key=lambda k: results[k].get('log_probability', float('-inf'))
        )
        best_result = results[best_key]

        path = plot_dir / f"best_trace_{self.args.test_aa}_{self.timestamp}.pdf"
        plot_fn(
            signal=best_result['z_normalized_signal'],
            segment_results=best_result['segment_results'],
            state_sequence=best_result.get('state_sequence', []),
            full_path=best_result.get('full_path', []),
            signal_key=f"BEST - {best_key}",
            save_path=str(path)
        )
        logger.info(f"Saved best trace plot: {path.name}")

    def _compute_profile_stats_for_plot(
        self,
        barycenters: Optional[Dict[str, List[np.ndarray]]],
        variance_collectors: Optional[Dict[str, SegmentVarianceCollector]]
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        if not barycenters or self.args.model_aa not in barycenters:
            return None

        profile_arrays = barycenters[self.args.model_aa]
        profile_stats = {}

        use_empirical = (
            self.args.variance_mode in ('segment', 'empirical')
            and variance_collectors
            and self.args.model_aa in variance_collectors
        )

        if use_empirical:
            empirical_variances = variance_collectors[self.args.model_aa].get_average_variances()
            for i in range(min(len(profile_arrays), len(empirical_variances))):
                seg_mean = float(np.mean(profile_arrays[i]))
                emp_var = empirical_variances[i]
                if isinstance(emp_var, (list, np.ndarray)):
                    emp_var = float(np.mean(emp_var))
                else:
                    emp_var = float(emp_var)
                profile_stats[str(i)] = (seg_mean, float(np.sqrt(emp_var)))
        else:
            for i, seg_array in enumerate(profile_arrays):
                seg_mean = float(np.mean(seg_array))
                seg_var = float(np.var(seg_array))
                if self.args.variance_mode == 'barycenter':
                    seg_var *= self.args.variance_scale
                profile_stats[str(i)] = (seg_mean, float(np.sqrt(seg_var)))

        return profile_stats

    def _generate_pairwise_matrix(
        self,
        results: Dict[str, Dict[str, Any]],
        classifier: Optional[HMMClassifier],
        plot_dir: Path
    ) -> None:
        """Generate pairwise classification matrix visualizations."""
        try:
            from vrhmm.visualization.pairwise_matrix import (
                plot_pairwise_classification_matrix,
                plot_pairwise_confusion_summary,
                plot_category_pairwise_matrix
            )
        except ImportError as e:
            logger.warning(f"Pairwise matrix visualization not available: {e}")
            return

        logger.info("Generating pairwise classification matrix...")

        try:
            if self.args.classification_mode == '20way':
                self._plot_20way_pairwise(results, classifier, plot_dir,
                                          plot_pairwise_classification_matrix,
                                          plot_pairwise_confusion_summary)
            else:
                path = plot_dir / f"pairwise_category_matrix_{self.args.classification_mode}_{self.timestamp}.pdf"
                plot_category_pairwise_matrix(
                    results,
                    self.args.classification_mode,
                    save_path=str(path)
                )
                logger.info(f"Saved category pairwise matrix to: {path.name}")
        except Exception as e:
            logger.error(f"Error generating pairwise matrix: {e}")

    def _plot_20way_pairwise(self, results, classifier, plot_dir,
                             plot_matrix_fn, plot_summary_fn) -> None:
        matrix_path = plot_dir / f"pairwise_accuracy_matrix_{self.timestamp}.pdf"
        matrix_df = plot_matrix_fn(
            results,
            classifier=classifier,
            metric='accuracy',
            save_path=str(matrix_path),
            title='Pairwise Amino Acid Classification Accuracy'
        )

        csv_path = plot_dir / f"pairwise_accuracy_matrix_{self.timestamp}.csv"
        matrix_df.to_csv(csv_path)
        logger.info(f"Saved pairwise matrix to: {matrix_path.name}")

        llr_path = plot_dir / f"pairwise_llr_matrix_{self.timestamp}.pdf"
        plot_matrix_fn(
            results,
            classifier=classifier,
            metric='llr',
            save_path=str(llr_path),
            title='Pairwise Log-Likelihood Ratio'
        )

        summary_path = plot_dir / f"pairwise_confusion_summary_{self.timestamp}.pdf"
        summary = plot_summary_fn(
            results,
            save_path=str(summary_path),
            top_n_confusions=15
        )

        logger.info(f"Pairwise classification summary:")
        logger.info(f"  Total pairs: {summary['total_pairs']}")
        logger.info(f"  Mean accuracy: {summary['mean_accuracy']:.3f}")
        logger.info(f"  Pairs below random: {summary['pairs_below_random']}")
        logger.info(f"  Pairs above 90%: {summary['pairs_above_90']}")

        
    # Reporting
    def _print_test_results(self, df: pd.DataFrame) -> None:
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