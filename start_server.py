#!/usr/bin/env python
"""
Simple script to start the API server
"""
import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add current directory to path so api module can be found
    sys.path.insert(0, os.getcwd())
    # Run the server
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)