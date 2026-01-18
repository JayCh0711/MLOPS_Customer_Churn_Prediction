"""
Run complete monitoring demo
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_full_demo():
    """Run all monitoring components"""
    
    print("="*70)
    print("                    MLOPS MONITORING DEMO")
    print("="*70)
    
    # 1. Data Drift Detection
    print("\n" + "="*70)
    print("STEP 1: Data Drift Detection")
    print("="*70)
    from monitoring.drift.drift_detector import run_drift_analysis
    run_drift_analysis()
    
    # 2. Data Profiling
    print("\n" + "="*70)
    print("STEP 2: Data Profiling")
    print("="*70)
    from monitoring.profiles.data_profiler import run_profiling
    run_profiling()
    
    # 3. Monitoring Service
    print("\n" + "="*70)
    print("STEP 3: Monitoring Service")
    print("="*70)
    from monitoring.monitoring_service import demo_monitoring
    demo_monitoring()
    
    # 4. Alerting System
    print("\n" + "="*70)
    print("STEP 4: Alerting System")
    print("="*70)
    from monitoring.alerts import demo_alerts
    demo_alerts()
    
    # 5. Dashboard
    print("\n" + "="*70)
    print("STEP 5: Monitoring Dashboard")
    print("="*70)
    from monitoring.reports.dashboard_data import generate_dashboard_data, create_html_dashboard
    data = generate_dashboard_data()
    dashboard_path = create_html_dashboard(data)
    print(f"Dashboard created: {dashboard_path}")
    
    # Summary
    print("\n" + "="*70)
    print("                    DEMO COMPLETE!")
    print("="*70)
    print("""
Monitoring outputs created:
  📊 monitoring/reports/     - Drift reports (HTML & JSON)
  📈 monitoring/profiles/    - Data profiles
  🚨 monitoring/alerts/      - Alert logs
  📋 monitoring/dashboard.html - Monitoring dashboard
  
To view the dashboard:
  1. Open monitoring/dashboard.html in a browser
  
To run scheduled monitoring:
  python monitoring/scheduled_monitoring.py
  
To integrate with API:
  The monitoring routes are available at /monitoring/*
    """)


if __name__ == "__main__":
    run_full_demo()