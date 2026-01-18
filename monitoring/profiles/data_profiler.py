"""
Data Profiling Module using WhyLogs
"""
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import whylogs as why
from whylogs.core.constraints import ConstraintsBuilder
from whylogs.core.constraints.factories import (
    greater_than_number,
    smaller_than_number,
    no_missing_values,
    is_in_range
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Data Profiling using WhyLogs
    """
    
    def __init__(
        self,
        profiles_dir: str = "monitoring/profiles",
        reference_profile_path: Optional[str] = None
    ):
        """
        Initialize data profiler
        
        Args:
            profiles_dir: Directory to save profiles
            reference_profile_path: Path to reference profile (optional)
        """
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        
        self.reference_profile = None
        if reference_profile_path and os.path.exists(reference_profile_path):
            self.reference_profile = why.read(reference_profile_path)
            logger.info(f"Loaded reference profile from {reference_profile_path}")
    
    def create_profile(
        self,
        data: pd.DataFrame,
        dataset_name: str = "production"
    ) -> Dict:
        """
        Create data profile for a dataset
        
        Args:
            data: DataFrame to profile
            dataset_name: Name for the dataset
            
        Returns:
            Dictionary with profile summary
        """
        logger.info(f"Creating profile for {len(data)} samples...")
        
        # Create profile
        result = why.log(data)
        profile = result.profile()
        
        # Get profile view
        profile_view = profile.view()
        
        # Save profile
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        profile_path = os.path.join(
            self.profiles_dir,
            f'profile_{dataset_name}_{timestamp}.bin'
        )
        result.writer("local").write(dest=profile_path)
        
        # Extract summary statistics
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'dataset_name': dataset_name,
            'num_samples': len(data),
            'num_features': len(data.columns),
            'profile_path': profile_path,
            'columns': {}
        }
        
        # Get column summaries
        columns_df = profile_view.to_pandas()
        
        for col in data.columns:
            if col in columns_df.index:
                col_stats = columns_df.loc[col].to_dict()
                
                # Extract key statistics
                summary['columns'][col] = {
                    'count': int(col_stats.get('counts/n', 0)),
                    'null_count': int(col_stats.get('counts/null', 0)),
                    'dtype': str(data[col].dtype)
                }
                
                # Add numerical stats if available
                if 'distribution/mean' in col_stats:
                    summary['columns'][col].update({
                        'mean': col_stats.get('distribution/mean'),
                        'stddev': col_stats.get('distribution/stddev'),
                        'min': col_stats.get('distribution/min'),
                        'max': col_stats.get('distribution/max')
                    })
        
        logger.info(f"Profile created and saved to {profile_path}")
        
        return summary
    
    def create_reference_profile(
        self,
        data: pd.DataFrame
    ) -> str:
        """
        Create and save reference profile
        
        Args:
            data: Reference data (typically training data)
            
        Returns:
            Path to saved reference profile
        """
        result = why.log(data)
        
        ref_path = os.path.join(self.profiles_dir, 'reference_profile.bin')
        result.writer("local").write(dest=ref_path)
        
        self.reference_profile = result.profile()
        
        logger.info(f"Reference profile saved to {ref_path}")
        
        return ref_path
    
    def compare_profiles(
        self,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Compare current data profile with reference
        
        Args:
            current_data: Current production data
            
        Returns:
            Dictionary with comparison results
        """
        if self.reference_profile is None:
            raise ValueError("No reference profile loaded. Create one first.")
        
        logger.info("Comparing profiles...")
        
        # Create current profile
        current_result = why.log(current_data)
        current_profile = current_result.profile()
        
        # Get views
        ref_view = self.reference_profile.view()
        cur_view = current_profile.view()
        
        ref_df = ref_view.to_pandas()
        cur_df = cur_view.to_pandas()
        
        # Compare statistics
        comparison = {
            'timestamp': datetime.utcnow().isoformat(),
            'reference_samples': int(ref_df['counts/n'].iloc[0]) if 'counts/n' in ref_df.columns else 0,
            'current_samples': len(current_data),
            'columns': {}
        }
        
        for col in current_data.columns:
            if col in ref_df.index and col in cur_df.index:
                ref_stats = ref_df.loc[col]
                cur_stats = cur_df.loc[col]
                
                comparison['columns'][col] = {
                    'reference': {},
                    'current': {},
                    'drift_indicators': {}
                }
                
                # Compare key metrics
                metrics_to_compare = [
                    'distribution/mean',
                    'distribution/stddev',
                    'distribution/min',
                    'distribution/max',
                    'counts/null'
                ]
                
                for metric in metrics_to_compare:
                    if metric in ref_stats and metric in cur_stats:
                        ref_val = ref_stats[metric]
                        cur_val = cur_stats[metric]
                        
                        metric_name = metric.split('/')[-1]
                        comparison['columns'][col]['reference'][metric_name] = ref_val
                        comparison['columns'][col]['current'][metric_name] = cur_val
                        
                        # Calculate drift indicator
                        if ref_val and ref_val != 0:
                            drift_pct = abs((cur_val - ref_val) / ref_val * 100)
                            comparison['columns'][col]['drift_indicators'][metric_name] = drift_pct
        
        return comparison
    
    def validate_constraints(
        self,
        data: pd.DataFrame,
        constraints_config: Dict
    ) -> Dict:
        """
        Validate data against constraints
        
        Args:
            data: Data to validate
            constraints_config: Dictionary of constraints per column
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data constraints...")
        
        result = why.log(data)
        profile = result.profile()
        
        # Build constraints
        builder = ConstraintsBuilder(dataset_profile_view=profile.view())
        
        for col, constraints in constraints_config.items():
            if col not in data.columns:
                continue
            
            if 'min' in constraints:
                builder.add_constraint(greater_than_number(col, constraints['min']))
            
            if 'max' in constraints:
                builder.add_constraint(smaller_than_number(col, constraints['max']))
            
            if constraints.get('no_nulls', False):
                builder.add_constraint(no_missing_values(col))
        
        # Run constraints
        constraints_obj = builder.build()
        report = constraints_obj.generate_constraints_report()
        
        validation_result = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_constraints': len(report),
            'passed': 0,
            'failed': 0,
            'results': []
        }
        
        for constraint_report in report:
            result_item = {
                'name': constraint_report.name,
                'passed': constraint_report.passed
            }
            validation_result['results'].append(result_item)
            
            if constraint_report.passed:
                validation_result['passed'] += 1
            else:
                validation_result['failed'] += 1
        
        validation_result['all_passed'] = validation_result['failed'] == 0
        
        logger.info(f"Constraints validation: {validation_result['passed']} passed, {validation_result['failed']} failed")
        
        return validation_result


def run_profiling():
    """Run data profiling demo"""
    
    print("="*60)
    print("DATA PROFILING WITH WHYLOGS")
    print("="*60)
    
    # Initialize profiler
    profiler = DataProfiler()
    
    # Load data
    train_data = pd.read_csv("data/processed/X_train.csv")
    test_data = pd.read_csv("data/processed/X_test.csv")
    
    # Create reference profile
    print("\n1. Creating reference profile from training data...")
    ref_path = profiler.create_reference_profile(train_data)
    print(f"   Reference profile saved to: {ref_path}")
    
    # Create current profile
    print("\n2. Creating profile for test data...")
    test_summary = profiler.create_profile(test_data, "test")
    print(f"   Samples profiled: {test_summary['num_samples']}")
    print(f"   Features profiled: {test_summary['num_features']}")
    
    # Compare profiles
    print("\n3. Comparing profiles...")
    comparison = profiler.compare_profiles(test_data)
    
    # Show some drift indicators
    print("\n   Top drift indicators:")
    for col, stats in list(comparison['columns'].items())[:5]:
        if stats['drift_indicators']:
            mean_drift = stats['drift_indicators'].get('mean', 0)
            print(f"   - {col}: mean drift = {mean_drift:.2f}%")
    
    # Validate constraints
    print("\n4. Validating data constraints...")
    constraints = {
        'tenure': {'min': 0, 'max': 100, 'no_nulls': True},
        'MonthlyCharges': {'min': 0, 'max': 200, 'no_nulls': True}
    }
    
    validation = profiler.validate_constraints(test_data, constraints)
    print(f"   Constraints passed: {validation['passed']}")
    print(f"   Constraints failed: {validation['failed']}")
    
    print("\n" + "="*60)
    print("Profiling complete! Check monitoring/profiles/ for saved profiles.")
    print("="*60)
    
    return test_summary, comparison, validation


if __name__ == "__main__":
    run_profiling()