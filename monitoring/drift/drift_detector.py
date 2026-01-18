"""
Data Drift Detection Module using Evidently
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataStabilityTestPreset
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Data Drift Detection using Evidently AI
    """
    
    def __init__(
        self,
        reference_data_path: str = "data/processed/X_train.csv",
        reports_dir: str = "monitoring/reports"
    ):
        """
        Initialize drift detector
        
        Args:
            reference_data_path: Path to reference (training) data
            reports_dir: Directory to save reports
        """
        self.reference_data = pd.read_csv(reference_data_path)
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        
        # Define column mapping
        self.column_mapping = ColumnMapping()
        
        # Identify column types
        self.numerical_columns = self.reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        self.categorical_columns = self.reference_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        logger.info(f"Initialized drift detector with {len(self.reference_data)} reference samples")
        logger.info(f"Numerical columns: {len(self.numerical_columns)}")
        logger.info(f"Categorical columns: {len(self.categorical_columns)}")
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True
    ) -> Dict:
        """
        Detect data drift between reference and current data
        
        Args:
            current_data: Current production data
            save_report: Whether to save HTML report
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info(f"Analyzing drift for {len(current_data)} samples...")
        
        # Create drift report
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
            DatasetMissingValuesMetric()
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        report_dict = drift_report.as_dict()
        
        # Parse results
        dataset_drift = report_dict['metrics'][0]['result']
        drift_table = report_dict['metrics'][1]['result']
        
        # Calculate drift summary
        drift_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'reference_samples': len(self.reference_data),
            'current_samples': len(current_data),
            'dataset_drift_detected': dataset_drift['dataset_drift'],
            'drift_share': dataset_drift['drift_share'],
            'number_of_drifted_columns': dataset_drift['number_of_drifted_columns'],
            'number_of_columns': dataset_drift['number_of_columns'],
            'drifted_columns': [],
            'column_drift_scores': {}
        }
        
        # Get per-column drift info
        for col_name, col_data in drift_table['drift_by_columns'].items():
            drift_summary['column_drift_scores'][col_name] = {
                'drift_detected': col_data['drift_detected'],
                'drift_score': col_data.get('drift_score', col_data.get('stattest_threshold', 0)),
                'stattest_name': col_data.get('stattest_name', 'unknown')
            }
            
            if col_data['drift_detected']:
                drift_summary['drifted_columns'].append(col_name)
        
        # Save HTML report
        if save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(
                self.reports_dir,
                f'drift_report_{timestamp}.html'
            )
            drift_report.save_html(report_path)
            drift_summary['report_path'] = report_path
            logger.info(f"Drift report saved to {report_path}")
        
        # Save JSON summary
        summary_path = os.path.join(
            self.reports_dir,
            f'drift_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(summary_path, 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        logger.info(f"Drift detected: {drift_summary['dataset_drift_detected']}")
        logger.info(f"Drifted columns: {drift_summary['number_of_drifted_columns']}/{drift_summary['number_of_columns']}")
        
        return drift_summary
    
    def run_data_quality_tests(
        self,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Run data quality tests on current data
        
        Args:
            current_data: Current production data
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running data quality tests...")
        
        # Create test suite
        test_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(lte=5),
            TestNumberOfRowsWithMissingValues(lte=len(current_data) * 0.1),
            TestNumberOfConstantColumns(eq=0),
            TestNumberOfDuplicatedRows(lte=len(current_data) * 0.05),
            TestNumberOfDuplicatedColumns(eq=0),
            TestNumberOfDriftedColumns(lt=len(current_data.columns) * 0.3)
        ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Get results
        results = test_suite.as_dict()
        
        # Parse test results
        test_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': []
        }
        
        for test in results['tests']:
            test_result = {
                'name': test['name'],
                'status': test['status'],
                'description': test.get('description', '')
            }
            test_summary['test_results'].append(test_result)
            
            if test['status'] == 'SUCCESS':
                test_summary['tests_passed'] += 1
            else:
                test_summary['tests_failed'] += 1
        
        test_summary['all_passed'] = test_summary['tests_failed'] == 0
        
        # Save test report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            self.reports_dir,
            f'quality_tests_{timestamp}.html'
        )
        test_suite.save_html(report_path)
        test_summary['report_path'] = report_path
        
        logger.info(f"Tests passed: {test_summary['tests_passed']}")
        logger.info(f"Tests failed: {test_summary['tests_failed']}")
        
        return test_summary
    
    def get_column_statistics(
        self,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Get statistical comparison between reference and current data
        
        Args:
            current_data: Current production data
            
        Returns:
            Dictionary with column statistics
        """
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'columns': {}
        }
        
        for col in self.numerical_columns:
            if col in current_data.columns:
                stats['columns'][col] = {
                    'reference': {
                        'mean': float(self.reference_data[col].mean()),
                        'std': float(self.reference_data[col].std()),
                        'min': float(self.reference_data[col].min()),
                        'max': float(self.reference_data[col].max()),
                        'median': float(self.reference_data[col].median())
                    },
                    'current': {
                        'mean': float(current_data[col].mean()),
                        'std': float(current_data[col].std()),
                        'min': float(current_data[col].min()),
                        'max': float(current_data[col].max()),
                        'median': float(current_data[col].median())
                    }
                }
                
                # Calculate percentage change
                ref_mean = stats['columns'][col]['reference']['mean']
                cur_mean = stats['columns'][col]['current']['mean']
                if ref_mean != 0:
                    stats['columns'][col]['mean_change_pct'] = (
                        (cur_mean - ref_mean) / abs(ref_mean) * 100
                    )
                else:
                    stats['columns'][col]['mean_change_pct'] = 0.0
        
        return stats


class ModelPerformanceMonitor:
    """
    Monitor model performance in production
    """
    
    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        baseline_metrics_path: str = "models/metrics/training_summary.json"
    ):
        """
        Initialize performance monitor
        
        Args:
            model_path: Path to trained model
            baseline_metrics_path: Path to baseline metrics
        """
        self.model = joblib.load(model_path)
        
        # Load baseline metrics
        with open(baseline_metrics_path, 'r') as f:
            training_summary = json.load(f)
            self.baseline_metrics = training_summary.get('best_metrics', {})
        
        logger.info("Initialized performance monitor")
        logger.info(f"Baseline metrics: {self.baseline_metrics}")
    
    def evaluate_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.1
    ) -> Dict:
        """
        Evaluate model performance and compare to baseline
        
        Args:
            X: Feature data
            y: True labels
            threshold: Alert threshold for performance degradation
            
        Returns:
            Dictionary with performance evaluation
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        current_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            current_metrics['roc_auc'] = roc_auc_score(y, y_prob)
        
        # Compare to baseline
        performance_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'samples_evaluated': len(X),
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'degradation_detected': False,
            'degraded_metrics': [],
            'metric_changes': {}
        }
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric_name, 0)
            
            if baseline_value > 0:
                change_pct = (current_value - baseline_value) / baseline_value * 100
            else:
                change_pct = 0
            
            performance_report['metric_changes'][metric_name] = {
                'baseline': baseline_value,
                'current': current_value,
                'change_pct': change_pct
            }
            
            # Check for degradation
            if change_pct < -threshold * 100:  # Negative change beyond threshold
                performance_report['degradation_detected'] = True
                performance_report['degraded_metrics'].append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct
                })
        
        # Log results
        logger.info(f"Performance evaluation complete for {len(X)} samples")
        if performance_report['degradation_detected']:
            logger.warning(f"Performance degradation detected in: {[m['metric'] for m in performance_report['degraded_metrics']]}")
        else:
            logger.info("No significant performance degradation detected")
        
        return performance_report


def run_drift_analysis():
    """Run drift analysis on test data (demo)"""
    
    print("="*60)
    print("DATA DRIFT ANALYSIS")
    print("="*60)
    
    # Initialize detector
    detector = DataDriftDetector()
    
    # Load test data as "current" data for demo
    current_data = pd.read_csv("data/processed/X_test.csv")
    
    # Simulate some drift by modifying data
    print("\n1. Analyzing original test data (no drift expected)...")
    results = detector.detect_drift(current_data)
    print(f"   Dataset drift detected: {results['dataset_drift_detected']}")
    print(f"   Drifted columns: {results['number_of_drifted_columns']}/{results['number_of_columns']}")
    
    # Simulate drift
    print("\n2. Simulating data drift...")
    drifted_data = current_data.copy()
    
    # Add noise to numerical columns
    for col in detector.numerical_columns[:3]:
        if col in drifted_data.columns:
            drifted_data[col] = drifted_data[col] * 1.5 + np.random.normal(0, 0.5, len(drifted_data))
    
    results_drifted = detector.detect_drift(drifted_data)
    print(f"   Dataset drift detected: {results_drifted['dataset_drift_detected']}")
    print(f"   Drifted columns: {results_drifted['number_of_drifted_columns']}/{results_drifted['number_of_columns']}")
    
    if results_drifted['drifted_columns']:
        print(f"   Columns with drift: {results_drifted['drifted_columns'][:5]}...")
    
    # Run quality tests
    print("\n3. Running data quality tests...")
    quality_results = detector.run_data_quality_tests(current_data)
    print(f"   Tests passed: {quality_results['tests_passed']}")
    print(f"   Tests failed: {quality_results['tests_failed']}")
    
    print("\n" + "="*60)
    print("Drift analysis complete! Check monitoring/reports/ for detailed reports.")
    print("="*60)
    
    return results, results_drifted, quality_results


if __name__ == "__main__":
    run_drift_analysis()