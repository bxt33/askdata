$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
& "D:\anaconda\envs\ai_agent\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload
