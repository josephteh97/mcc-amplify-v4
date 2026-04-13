@echo off
echo Setting up Amplify AI Environment...

cd /d "%~dp0..\backend"
if exist venv (
    echo Virtual environment found.
) else (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
pip install transformers qwen-vl-utils accelerate

echo Setup complete.
pause
