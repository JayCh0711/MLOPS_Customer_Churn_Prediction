"""
Monitoring Service for Production API
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log predictions for monitoring
    """
    
    def __init__(
        self,
        log_dir: str = "monitoring/predictions",
        max_buffer_size: int = 1000,
        flush_interval: int = 300  # 5 minutes
    ):
        """
        Initialize prediction logger
        
        Args:
            log_dir: Directory to save prediction logs
            max_buffer_size: Max predictions to buffer before flushing
            flush_interval: Seconds between automatic flushes
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.buffer = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self.last_flush = datetime.now()
        
        self._lock = threading.Lock()
        
        logger.info(f"Prediction logger initialized. Log dir: {log_dir}")
    
    def log_prediction(
        self,
        features: Dict,
        prediction: int,
        probability: float,
        customer_id: Optional[str] = None,
        actual_label: Optional[int] = None
    ):
        """
        Log a single prediction
        
        Args:
            features: Input features
            prediction: Model prediction
            probability: Prediction probability
            customer_id: Optional customer ID
            actual_label: Optional actual label (if known)
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'customer_id': customer_id,
            'features': features,
            'prediction': prediction,
            'probability': probability,
            'actual_label': actual_label
        }
        
        with self._lock:
            self.buffer.append(log_entry)
            
            # Auto-flush if buffer is full
            if len(self.buffer) >= self.max_buffer_size:
                self._flush()
            
            # Auto-flush based on time
            if (datetime.now() - self.last_flush).seconds > self.flush_interval:
                self._flush()
    
    def _flush(self):
        """Flush buffer to disk"""
        if not self.buffer:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'predictions_{timestamp}.json')
        
        with open(log_file, 'w') as f:
            json.dump(list(self.buffer), f, indent=2, default=str)
        
        logger.info(f"Flushed {len(self.buffer)} predictions to {log_file}")
        
        self.buffer.clear()
        self.last_flush = datetime.now()
    
    def flush(self):
        """Public flush method"""
        with self._lock:
            self._flush()
    
    def get_recent_predictions(self, n: int = 100) -> List[Dict]:
        """Get recent predictions from buffer"""
        with self._lock:
            return list(self.buffer)[-n:]


class MonitoringService:
    """
    Main monitoring service for the ML API
    """
    
    def __init__(
        self,
        reference_data_path: str = "data/processed/X_train.csv",
        monitoring_dir: str = "monitoring"
    ):
        """
        Initialize monitoring service
        
        Args:
            reference_data_path: Path to reference data
            monitoring_dir: Base directory for monitoring outputs
        """
        self.monitoring_dir = monitoring_dir
        self.reference_data = pd.read_csv(reference_data_path)
        
        # Initialize components
        self.prediction_logger = PredictionLogger(
            log_dir=os.path.join(monitoring_dir, "predictions")
        )
        
        # Metrics storage
        self.metrics = {
            'total_predictions': 0,
            'predictions_by_hour': {},
            'churn_predictions': 0,
            'no_churn_predictions': 0,
            'average_probability': 0,
            'high_risk_count': 0,  # probability > 0.7
            'low_risk_count': 0,   # probability < 0.3
        }
        
        self.alerts = []
        self._lock = threading.Lock()
        
        # Thresholds
        self.thresholds = {
            'drift_alert': 0.3,  # Alert if > 30% columns drift
            'performance_drop': 0.1,  # Alert if performance drops > 10%
            'high_risk_ratio': 0.5,  # Alert if > 50% predictions are high risk
        }
        
        logger.info("Monitoring service initialized")
    
    def record_prediction(
        self,
        features: Dict,
        prediction: int,
        probability: float,
        customer_id: Optional[str] = None
    ):
        """
        Record a prediction for monitoring
        
        Args:
            features: Input features
            prediction: Model prediction
            probability: Prediction probability
            customer_id: Optional customer ID
        """
        # Log prediction
        self.prediction_logger.log_prediction(
            features=features,
            prediction=prediction,
            probability=probability,
            customer_id=customer_id
        )
        
        # Update metrics
        with self._lock:
            self.metrics['total_predictions'] += 1
            
            hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
            self.metrics['predictions_by_hour'][hour_key] = \
                self.metrics['predictions_by_hour'].get(hour_key, 0) + 1
            
            if prediction == 1:
                self.metrics['churn_predictions'] += 1
            else:
                self.metrics['no_churn_predictions'] += 1
            
            # Running average
            n = self.metrics['total_predictions']
            self.metrics['average_probability'] = (
                (self.metrics['average_probability'] * (n - 1) + probability) / n
            )
            
            if probability > 0.7:
                self.metrics['high_risk_count'] += 1
            elif probability < 0.3:
                self.metrics['low_risk_count'] += 1
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check for monitoring alerts"""
        with self._lock:
            total = self.metrics['total_predictions']
            
            if total < 10:
                return  # Not enough data
            
            # Check high risk ratio
            high_risk_ratio = self.metrics['high_risk_count'] / total
            if high_risk_ratio > self.thresholds['high_risk_ratio']:
                self._add_alert(
                    'HIGH_RISK_RATIO',
                    f'High risk predictions ratio ({high_risk_ratio:.2%}) exceeds threshold'
                )
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': alert_type,
            'message': message
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self._lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            total = metrics['total_predictions']
            if total > 0:
                metrics['churn_rate'] = metrics['churn_predictions'] / total
                metrics['high_risk_rate'] = metrics['high_risk_count'] / total
                metrics['low_risk_rate'] = metrics['low_risk_count'] / total
            
            return metrics
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict]:
        """Get alerts, optionally filtered by time"""
        if since is None:
            return self.alerts.copy()
        
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > since
        ]
    
    def run_drift_check(self) -> Dict:
        """
        Run drift check on recent predictions
        
        Returns:
            Drift analysis results
        """
        from monitoring.drift_detector import DataDriftDetector
        
        # Get recent prediction features
        recent = self.prediction_logger.get_recent_predictions(100)
        
        if len(recent) < 50:
            return {'status': 'insufficient_data', 'samples': len(recent)}
        
        # Convert to DataFrame
        features_list = [p['features'] for p in recent]
        current_data = pd.DataFrame(features_list)
        
        # Run drift detection
        detector = DataDriftDetector()
        drift_results = detector.detect_drift(current_data, save_report=True)
        
        # Check for alert
        if drift_results['dataset_drift_detected']:
            self._add_alert(
                'DATA_DRIFT',
                f"Data drift detected in {drift_results['number_of_drifted_columns']} columns"
            )
        
        return drift_results
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Monitoring report dictionary
        """
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': self.get_metrics(),
            'recent_alerts': self.get_alerts(
                since=datetime.now() - timedelta(hours=24)
            ),
            'status': 'healthy'
        }
        
        # Determine overall status
        if len(report['recent_alerts']) > 0:
            report['status'] = 'warning'
        
        critical_alerts = [
            a for a in report['recent_alerts']
            if a['type'] in ['DATA_DRIFT', 'PERFORMANCE_DEGRADATION']
        ]
        if len(critical_alerts) > 0:
            report['status'] = 'critical'
        
        # Save report
        report_path = os.path.join(
            self.monitoring_dir,
            'reports',
            f'monitoring_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitoring report saved to {report_path}")
        
        return report


# Global monitoring service instance
_monitoring_service = None


def get_monitoring_service() -> MonitoringService:
    """Get or create monitoring service singleton"""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    
    return _monitoring_service


def demo_monitoring():
    """Demo monitoring service"""
    
    print("="*60)
    print("MONITORING SERVICE DEMO")
    print("="*60)
    
    service = get_monitoring_service()
    
    # Simulate some predictions
    print("\n1. Simulating predictions...")
    
    np.random.seed(42)
    
    for i in range(100):
        # Simulate features
        features = {
            'tenure': np.random.randint(0, 72),
            'MonthlyCharges': np.random.uniform(20, 100),
            'TotalCharges': np.random.uniform(100, 5000),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'])
        }
        
        # Simulate prediction
        prediction = np.random.choice([0, 1], p=[0.7, 0.3])
        probability = np.random.beta(2, 5) if prediction == 0 else np.random.beta(5, 2)
        
        service.record_prediction(
            features=features,
            prediction=prediction,
            probability=probability,
            customer_id=f"CUST{i:04d}"
        )
    
    print(f"   Recorded 100 predictions")
    
    # Get metrics
    print("\n2. Current Metrics:")
    metrics = service.get_metrics()
    print(f"   Total predictions: {metrics['total_predictions']}")
    print(f"   Churn rate: {metrics.get('churn_rate', 0):.2%}")
    print(f"   Average probability: {metrics['average_probability']:.4f}")
    print(f"   High risk count: {metrics['high_risk_count']}")
    
    # Check for alerts
    print("\n3. Active Alerts:")
    alerts = service.get_alerts()
    if alerts:
        for alert in alerts:
            print(f"   - [{alert['type']}] {alert['message']}")
    else:
        print("   No alerts")
    
    # Generate report
    print("\n4. Generating monitoring report...")
    report = service.generate_report()
    print(f"   Status: {report['status']}")
    print(f"   Report saved to monitoring/reports/")
    
    print("\n" + "="*60)
    print("Monitoring demo complete!")
    print("="*60)
    
    return service, metrics, report


if __name__ == "__main__":
    demo_monitoring()