@echo off
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo   Drug Side Effect Prediction - Startup Script
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Step 1: Virtual Environment Setup
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b
    )
)

:: Step 2: Install Dependencies
echo [INFO] Activating virtual environment and checking dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Step 3: Check for Model Files
if not exist "model\model.h5" (
    echo [WARNING] Model file not found. Training model now...
    cd model
    python train_model.py
    cd ..
) else (
    echo [INFO] Model files found. Skipping training.
)

:: Step 4: Run Application
echo [SUCCESS] Starting the Flask server...
echo Access the app at: http://127.0.0.1:5000/
python app.py

pause
