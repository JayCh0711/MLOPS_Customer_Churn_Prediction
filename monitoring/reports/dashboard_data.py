"""
Dashboard Data Generator for Monitoring
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List


def generate_dashboard_data(
    monitoring_dir: str = "monitoring",
    days: int = 7
) -> Dict:
    """
    Generate data for monitoring dashboard
    
    Args:
        monitoring_dir: Monitoring directory
        days: Number of days to include
        
    Returns:
        Dashboard data dictionary
    """
    
    dashboard_data = {
        'generated_at': datetime.utcnow().isoformat(),
        'time_range_days': days,
        'summary': {},
        'charts': {}
    }
    
    # Generate sample time series data
    dates = pd.date_range(
        end=datetime.now(),
        periods=days * 24,
        freq='h'
    )
    
    # Predictions over time
    np.random.seed(42)
    predictions_data = {
        'timestamps': [d.isoformat() for d in dates],
        'total_predictions': np.cumsum(np.random.poisson(50, len(dates))).tolist(),
        'churn_predictions': np.cumsum(np.random.poisson(15, len(dates))).tolist(),
        'hourly_predictions': np.random.poisson(50, len(dates)).tolist()
    }
    dashboard_data['charts']['predictions_timeline'] = predictions_data
    
    # Drift scores over time (daily)
    daily_dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    drift_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in daily_dates],
        'drift_scores': (np.random.beta(2, 5, len(daily_dates)) * 0.5).tolist(),
        'threshold': 0.3
    }
    dashboard_data['charts']['drift_timeline'] = drift_data
    
    # Model performance over time
    performance_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in daily_dates],
        'accuracy': (0.78 + np.random.normal(0, 0.02, len(daily_dates))).clip(0.7, 0.85).tolist(),
        'f1_score': (0.65 + np.random.normal(0, 0.03, len(daily_dates))).clip(0.55, 0.75).tolist(),
        'roc_auc': (0.82 + np.random.normal(0, 0.02, len(daily_dates))).clip(0.75, 0.88).tolist()
    }
    dashboard_data['charts']['performance_timeline'] = performance_data
    
    # Feature drift by column
    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
                     'InternetService', 'PaymentMethod', 'SeniorCitizen']
    drift_scores = np.random.beta(2, 8, len(feature_names)).tolist()
    drift_detected = [bool(s > 0.15) for s in np.random.beta(2, 8, len(feature_names))]
    feature_drift = {
        'features': feature_names,
        'drift_scores': drift_scores,
        'drift_detected': drift_detected
    }
    dashboard_data['charts']['feature_drift'] = feature_drift
    
    # Prediction distribution
    prediction_dist = {
        'labels': ['No Churn', 'Churn'],
        'values': [7250, 2750],
        'percentages': [72.5, 27.5]
    }
    dashboard_data['charts']['prediction_distribution'] = prediction_dist
    
    # Risk level distribution
    risk_dist = {
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'values': [4500, 3000, 2500],
        'percentages': [45, 30, 25]
    }
    dashboard_data['charts']['risk_distribution'] = risk_dist
    
    # Summary metrics
    dashboard_data['summary'] = {
        'total_predictions_24h': int(np.random.poisson(50) * 24),
        'churn_rate_24h': 0.275,
        'drift_detected': False,
        'model_accuracy': 0.782,
        'avg_latency_ms': 45,
        'active_alerts': 2,
        'last_model_update': (datetime.now() - timedelta(days=5)).isoformat()
    }
    
    # Save dashboard data
    output_path = os.path.join(monitoring_dir, 'dashboard_data.json')
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    return dashboard_data


def create_html_dashboard(
    data: Dict,
    output_path: str = "monitoring/dashboard.html"
) -> str:
    """
    Create simple HTML dashboard
    
    Args:
        data: Dashboard data dictionary
        output_path: Path to save HTML file
        
    Returns:
        Path to generated HTML file
    """
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 5px;
        }}
        .header p {{
            opacity: 0.8;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-card .change {{
            font-size: 12px;
            margin-top: 5px;
        }}
        .change.positive {{ color: #4CAF50; }}
        .change.negative {{ color: #f44336; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-card h3 {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #333;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-healthy {{ background: #E8F5E9; color: #4CAF50; }}
        .status-warning {{ background: #FFF3E0; color: #FF9800; }}
        .status-critical {{ background: #FFEBEE; color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 MLOps Monitoring Dashboard</h1>
        <p>Customer Churn Prediction Model | Last updated: {data['generated_at']}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Predictions (24h)</h3>
            <div class="value">{data['summary']['total_predictions_24h']:,}</div>
            <div class="change positive">↑ 12% vs yesterday</div>
        </div>
        <div class="metric-card">
            <h3>Churn Rate</h3>
            <div class="value">{data['summary']['churn_rate_24h']:.1%}</div>
            <div class="change negative">↑ 2% vs baseline</div>
        </div>
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <div class="value">{data['summary']['model_accuracy']:.1%}</div>
            <div class="change positive">Stable</div>
        </div>
        <div class="metric-card">
            <h3>Avg Latency</h3>
            <div class="value">{data['summary']['avg_latency_ms']}ms</div>
            <div class="change positive">↓ 5ms vs last week</div>
        </div>
        <div class="metric-card">
            <h3>Drift Status</h3>
            <div class="value">
                <span class="status-badge status-healthy">No Drift</span>
            </div>
        </div>
        <div class="metric-card">
            <h3>Active Alerts</h3>
            <div class="value">{data['summary']['active_alerts']}</div>
            <div class="change">2 warnings, 0 critical</div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-card">
            <h3>📈 Predictions Over Time</h3>
            <canvas id="predictionsChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>🎯 Model Performance</h3>
            <canvas id="performanceChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>📊 Prediction Distribution</h3>
            <canvas id="distributionChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>⚠️ Feature Drift Scores</h3>
            <canvas id="driftChart"></canvas>
        </div>
    </div>
    
    <script>
        // Predictions Chart
        const predData = {json.dumps(data['charts']['predictions_timeline'])};
        new Chart(document.getElementById('predictionsChart'), {{
            type: 'line',
            data: {{
                labels: predData.timestamps.slice(-48).map(t => new Date(t).toLocaleTimeString()),
                datasets: [{{
                    label: 'Hourly Predictions',
                    data: predData.hourly_predictions.slice(-48),
                    borderColor: '#667eea',
                    tension: 0.4
                }}]
            }},
            options: {{ responsive: true }}
        }});
        
        // Performance Chart
        const perfData = {json.dumps(data['charts']['performance_timeline'])};
        new Chart(document.getElementById('performanceChart'), {{
            type: 'line',
            data: {{
                labels: perfData.dates,
                datasets: [
                    {{ label: 'Accuracy', data: perfData.accuracy, borderColor: '#4CAF50' }},
                    {{ label: 'F1 Score', data: perfData.f1_score, borderColor: '#2196F3' }},
                    {{ label: 'ROC-AUC', data: perfData.roc_auc, borderColor: '#FF9800' }}
                ]
            }},
            options: {{ responsive: true }}
        }});
        
        // Distribution Chart
        const distData = {json.dumps(data['charts']['prediction_distribution'])};
        new Chart(document.getElementById('distributionChart'), {{
            type: 'doughnut',
            data: {{
                labels: distData.labels,
                datasets: [{{
                    data: distData.values,
                    backgroundColor: ['#4CAF50', '#f44336']
                }}]
            }},
            options: {{ responsive: true }}
        }});
        
        // Drift Chart
        const driftData = {json.dumps(data['charts']['feature_drift'])};
        new Chart(document.getElementById('driftChart'), {{
            type: 'bar',
            data: {{
                labels: driftData.features,
                datasets: [{{
                    label: 'Drift Score',
                    data: driftData.drift_scores,
                    backgroundColor: driftData.drift_scores.map(s => s > 0.15 ? '#f44336' : '#4CAF50')
                }}]
            }},
            options: {{ 
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, max: 0.5 }} }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


if __name__ == "__main__":
    print("Generating dashboard data...")
    data = generate_dashboard_data()
    
    print("Creating HTML dashboard...")
    path = create_html_dashboard(data)
    
    print(f"\nDashboard created: {path}")
    print("Open in browser to view the monitoring dashboard.")