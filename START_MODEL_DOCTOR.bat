@echo off
set "ROOT=%~dp0"
start "Model Doctor Backend" cmd /k ""%ROOT%scripts\start_backend.bat""
start "Model Doctor Frontend" cmd /k ""%ROOT%scripts\start_frontend.bat""
echo.
echo Model Doctor backend and frontend are starting in separate windows.
echo Open the Vite URL shown by the frontend window.
pause
