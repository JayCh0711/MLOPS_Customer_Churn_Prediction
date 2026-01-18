"""
Script to run the FastAPI application
"""
import uvicorn
import os

if __name__ == "__main__":
    # Disable reload during testing to avoid interference
    reload_enabled = os.getenv("DISABLE_RELOAD", "false").lower() != "true"

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
        log_level="info"
    )