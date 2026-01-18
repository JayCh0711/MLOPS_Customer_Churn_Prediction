"""
Scheduled Monitoring Jobs
"""
import schedule
import time
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.monitoring_service import get_monitoring_service
from monitoring.drift_detector import DataDriftDetector, ModelPerformanceMonitor
from monitoring.data_profiler import DataProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_hourly_metrics():
    """Collect and log hourly metrics"""
    logger.info("Running hourly metrics collection...")
    
    try:
        service = get_monitoring_service()
        metrics = service.get_metrics()
        
        logger.info(f"Hourly metrics - Total predictions: {metrics['total_predictions']}, "
                   f"Churn rate: {metrics.get('churn_rate', 0):.2%}")
        
        # Flush prediction logs
        service.prediction_logger.flush()
        
    except Exception as e:
        logger.error(f"Error in hourly metrics: {e}")


def run_daily_drift_check():
    """Run daily drift detection"""
    logger.info("Running daily drift check...")
    
    try:
        service = get_monitoring_service()
        result = service.run_drift_check()
        
        if result.get('status') != 'insufficient_data':
            if result['dataset_drift_detected']:
                logger.warning(f"DRIFT DETECTED: {result['number_of_drifted_columns']} columns drifted")
            else:
                logger.info("No significant drift detected")
        else:
            logger.info("Insufficient data for drift check")
            
    except Exception as e:
        logger.error(f"Error in daily drift check: {e}")


def run_daily_report():
    """Generate daily monitoring report"""
    logger.info("Generating daily report...")
    
    try:
        service = get_monitoring_service()
        report = service.generate_report()
        
        logger.info(f"Daily report generated - Status: {report['status']}, "
                   f"Alerts: {len(report['recent_alerts'])}")
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")


def run_weekly_profiling():
    """Run weekly data profiling"""
    logger.info("Running weekly data profiling...")
    
    try:
        # Get recent prediction data
        service = get_monitoring_service()
        recent = service.prediction_logger.get_recent_predictions(1000)
        
        if len(recent) < 100:
            logger.info("Insufficient data for profiling")
            return
        
        import pandas as pd
        features_list = [p['features'] for p in recent]
        current_data = pd.DataFrame(features_list)
        
        profiler = DataProfiler()
        profile = profiler.create_profile(current_data, "weekly")
        
        logger.info(f"Weekly profile created with {profile['num_samples']} samples")
        
    except Exception as e:
        logger.error(f"Error in weekly profiling: {e}")


def setup_scheduler():
    """Setup scheduled jobs"""
    
    # Hourly jobs
    schedule.every().hour.do(run_hourly_metrics)
    
    # Daily jobs
    schedule.every().day.at("06:00").do(run_daily_drift_check)
    schedule.every().day.at("07:00").do(run_daily_report)
    
    # Weekly jobs
    schedule.every().monday.at("08:00").do(run_weekly_profiling)
    
    logger.info("Scheduler setup complete")
    logger.info(f"Scheduled jobs: {len(schedule.jobs)}")


def run_scheduler():
    """Run the scheduler"""
    setup_scheduler()
    
    logger.info("Starting monitoring scheduler...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    # For testing, run all jobs once
    print("Running all monitoring jobs once for testing...")
    
    run_hourly_metrics()
    run_daily_drift_check()
    run_daily_report()
    # run_weekly_profiling()  # Skip in test mode
    
    print("\nTo run scheduler continuously, call run_scheduler()")