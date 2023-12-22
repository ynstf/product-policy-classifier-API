#!/bin/bash
APP_PORT=${PORT:-1111}
cd /app/
/opt/venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port ${APP_PORT} 
