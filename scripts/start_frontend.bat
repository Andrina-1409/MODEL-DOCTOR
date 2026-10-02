@echo off
cd /d "%~dp0..\frontend"
npm.cmd install
npm.cmd run dev
pause
