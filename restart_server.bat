@echo off
REM Kill any existing uvicorn on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
)
timeout /t 3 /nobreak > nul
set APP_DEBUG=1
cd /d d:\Ask_Pera\PERAIA
start /B "" d:\Ask_Pera\PERAIA\venv\Scripts\python.exe -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
echo Server starting on http://localhost:8000
echo Debug endpoint: http://localhost:8000/debug/index
echo Frontend: http://localhost:3000
timeout /t 10 /nobreak > nul
echo.
echo Testing server...
d:\Ask_Pera\PERAIA\venv\Scripts\python.exe -c "import requests; r=requests.get('http://localhost:8000/debug/index',timeout=10); print('STATUS:', r.status_code); print(r.text[:400])"
