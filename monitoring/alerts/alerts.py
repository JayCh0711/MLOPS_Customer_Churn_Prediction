"""
Alerting System for MLOps Monitoring
"""
import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    DATA_DRIFT = "data_drift"
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    PREDICTION_ANOMALY = "prediction_anomaly"


@dataclass
class Alert:
    """Alert data class"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class AlertManager:
    """
    Manage and dispatch alerts
    """
    
    def __init__(
        self,
        alerts_dir: str = "monitoring/alerts",
        config: Optional[Dict] = None
    ):
        """
        Initialize alert manager
        
        Args:
            alerts_dir: Directory to store alerts
            config: Configuration dictionary
        """
        self.alerts_dir = alerts_dir
        os.makedirs(alerts_dir, exist_ok=True)
        
        self.config = config or {}
        self.alert_history: List[Alert] = []
        self.alert_counter = 0
        
        # Load configuration from environment
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('ALERT_EMAIL_USERNAME'),
            'password': os.getenv('ALERT_EMAIL_PASSWORD'),
            'recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
        }
        
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.teams_webhook = os.getenv('TEAMS_WEBHOOK_URL')
        
        logger.info("Alert manager initialized")
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Alert:
        """
        Create a new alert
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Created Alert object
        """
        self.alert_counter += 1
        
        alert = Alert(
            alert_id=f"ALT-{datetime.now().strftime('%Y%m%d')}-{self.alert_counter:04d}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata
        )
        
        self.alert_history.append(alert)
        
        # Save alert
        self._save_alert(alert)
        
        # Dispatch alert
        self._dispatch_alert(alert)
        
        return alert
    
    def _save_alert(self, alert: Alert):
        """Save alert to file"""
        date_str = datetime.now().strftime('%Y%m%d')
        alerts_file = os.path.join(self.alerts_dir, f'alerts_{date_str}.json')
        
        # Load existing alerts
        existing_alerts = []
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r') as f:
                existing_alerts = json.load(f)
        
        # Add new alert
        existing_alerts.append(alert.to_dict())
        
        # Save
        with open(alerts_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)
    
    def _dispatch_alert(self, alert: Alert):
        """Dispatch alert to configured channels"""
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }.get(alert.severity, logging.INFO)
        
        logger.log(log_level, f"ALERT [{alert.severity.value}] {alert.title}: {alert.message}")
        
        # Send to Slack
        if self.slack_webhook:
            self._send_slack_alert(alert)
        
        # Send to Teams
        if self.teams_webhook:
            self._send_teams_alert(alert)
        
        # Send email for critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            self._send_email_alert(alert)
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        try:
            color = {
                AlertSeverity.INFO: '#36a64f',
                AlertSeverity.WARNING: '#ffcc00',
                AlertSeverity.CRITICAL: '#ff0000'
            }.get(alert.severity, '#808080')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"[{alert.severity.value.upper()}] {alert.title}",
                    'text': alert.message,
                    'fields': [
                        {'title': 'Alert Type', 'value': alert.alert_type.value, 'short': True},
                        {'title': 'Alert ID', 'value': alert.alert_id, 'short': True}
                    ],
                    'footer': 'MLOps Monitoring',
                    'ts': datetime.now().timestamp()
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_teams_alert(self, alert: Alert):
        """Send alert to Microsoft Teams"""
        try:
            color = {
                AlertSeverity.INFO: '36a64f',
                AlertSeverity.WARNING: 'ffcc00',
                AlertSeverity.CRITICAL: 'ff0000'
            }.get(alert.severity, '808080')
            
            payload = {
                '@type': 'MessageCard',
                '@context': 'http://schema.org/extensions',
                'themeColor': color,
                'summary': alert.title,
                'sections': [{
                    'activityTitle': f"[{alert.severity.value.upper()}] {alert.title}",
                    'facts': [
                        {'name': 'Alert Type', 'value': alert.alert_type.value},
                        {'name': 'Alert ID', 'value': alert.alert_id},
                        {'name': 'Timestamp', 'value': alert.timestamp}
                    ],
                    'text': alert.message
                }]
            }
            
            response = requests.post(self.teams_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Teams alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.email_config['username'] or not self.email_config['recipients']:
            logger.warning("Email not configured, skipping email alert")
            return
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(self.email_config['recipients'])
            
            html = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'};">
                    {alert.title}
                </h2>
                <p><strong>Alert ID:</strong> {alert.alert_id}</p>
                <p><strong>Type:</strong> {alert.alert_type.value}</p>
                <p><strong>Severity:</strong> {alert.severity.value}</p>
                <p><strong>Timestamp:</strong> {alert.timestamp}</p>
                <hr>
                <p>{alert.message}</p>
                {f'<pre>{json.dumps(alert.metadata, indent=2)}</pre>' if alert.metadata else ''}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alerts with optional filtering
        
        Args:
            severity: Filter by severity
            alert_type: Filter by type
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        alerts = self.alert_history.copy()
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts[-limit:]


# Convenience functions
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get or create alert manager singleton"""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
    
    return _alert_manager


def send_drift_alert(
    drift_share: float,
    drifted_columns: List[str],
    severity: AlertSeverity = AlertSeverity.WARNING
):
    """Send data drift alert"""
    manager = get_alert_manager()
    
    manager.create_alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=severity,
        title="Data Drift Detected",
        message=f"Data drift detected with {drift_share:.1%} of columns drifted.",
        metadata={
            'drift_share': drift_share,
            'drifted_columns': drifted_columns
        }
    )


def send_performance_alert(
    metric_name: str,
    baseline_value: float,
    current_value: float,
    change_pct: float
):
    """Send performance degradation alert"""
    manager = get_alert_manager()
    
    severity = AlertSeverity.CRITICAL if change_pct < -20 else AlertSeverity.WARNING
    
    manager.create_alert(
        alert_type=AlertType.MODEL_PERFORMANCE,
        severity=severity,
        title=f"Model Performance Degradation: {metric_name}",
        message=f"{metric_name} dropped from {baseline_value:.4f} to {current_value:.4f} ({change_pct:+.1f}%)",
        metadata={
            'metric_name': metric_name,
            'baseline_value': baseline_value,
            'current_value': current_value,
            'change_pct': change_pct
        }
    )


def demo_alerts():
    """Demo alert system"""
    
    print("="*60)
    print("ALERTING SYSTEM DEMO")
    print("="*60)
    
    manager = get_alert_manager()
    
    # Create sample alerts
    print("\n1. Creating sample alerts...")
    
    alert1 = manager.create_alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.WARNING,
        title="Data Drift Detected",
        message="5 out of 20 features show significant drift",
        metadata={'drift_share': 0.25, 'drifted_columns': ['tenure', 'MonthlyCharges']}
    )
    print(f"   Created: {alert1.alert_id}")
    
    alert2 = manager.create_alert(
        alert_type=AlertType.MODEL_PERFORMANCE,
        severity=AlertSeverity.CRITICAL,
        title="F1 Score Degradation",
        message="F1 score dropped from 0.75 to 0.62 (-17.3%)",
        metadata={'metric': 'f1_score', 'baseline': 0.75, 'current': 0.62}
    )
    print(f"   Created: {alert2.alert_id}")
    
    alert3 = manager.create_alert(
        alert_type=AlertType.SYSTEM_HEALTH,
        severity=AlertSeverity.INFO,
        title="High Prediction Volume",
        message="Received 10,000 predictions in the last hour",
        metadata={'count': 10000}
    )
    print(f"   Created: {alert3.alert_id}")
    
    # List alerts
    print("\n2. All alerts:")
    for alert in manager.get_alerts():
        print(f"   [{alert.severity.value}] {alert.title}")
    
    # Filter by severity
    print("\n3. Critical alerts only:")
    for alert in manager.get_alerts(severity=AlertSeverity.CRITICAL):
        print(f"   {alert.alert_id}: {alert.title}")
    
    print("\n" + "="*60)
    print("Alerting demo complete! Alerts saved to monitoring/alerts/")
    print("="*60)
    
    return manager


if __name__ == "__main__":
    demo_alerts()