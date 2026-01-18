#!/usr/bin/env python
"""
Script to run MLflow UI server
"""
import os
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Run the server
from mlflow.server import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)