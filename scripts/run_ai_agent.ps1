$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
& "D:\anaconda\envs\ai_agent\python.exe" -m streamlit run main.py --server.port 8501 --server.address 127.0.0.1
