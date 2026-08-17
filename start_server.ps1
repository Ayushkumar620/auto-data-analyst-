Set-Location c:/users/ayush/Desktop/auto-data-analyst
& 'c:/users/ayush/Desktop/auto-data-analyst/venv/Scripts/Activate.ps1'
& 'c:/users/ayush/Desktop/auto-data-analyst/venv/Scripts/uvicorn.exe' backend.app.main:app --host 0.0.0.0 --port 8000 --reload