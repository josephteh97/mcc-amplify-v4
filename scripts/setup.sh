#!/bin/bash
echo "Setting up Amplify AI Environment..."

cd "$(dirname "$0")/../backend" || exit

if [ -d "venv" ]; then
    echo "Virtual environment found."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
pip install transformers qwen-vl-utils accelerate

echo "Setup complete."
