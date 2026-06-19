#!/bin/bash

# ==============================================================================
# Quantangle-SAT: Environment Setup Script
# Authors: Shang-Wei LIN, Ji-Qing YAN, Yean-Ru CHEN, Zhe HOU, David Sanán
# Target: SAT 2026 Submission
# ==============================================================================

echo "------------------------------------------------"
echo "🚀 Quantangle-SAT: Initializing Environment"
echo "------------------------------------------------"

# 1. Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# 2. Check for the 'venv' module (Commonly missing on Ubuntu/Debian)
# We try to run the help command for venv to see if the module exists
python3 -m venv --help &> /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Python 'venv' module is not installed."
    echo "   On Ubuntu/Debian, please run: sudo apt update && sudo apt install python3-venv"
    exit 1
fi

# 3. Verify requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: 'requirements.txt' not found in the current directory."
    exit 1
fi

# 4. Create Virtual Environment
if [ ! -d "quantangle_venv" ]; then
    echo "📦 Creating virtual environment 'quantangle_venv'..."
    python3 -m venv quantangle_venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "ℹ️  Virtual environment 'quantangle_venv' already exists."
fi

# 5. Initialize Required Directories
# Creating these prevents the Solver from crashing on first run
echo "📁 Initializing project structure..."
mkdir -p experiments

# 6. Activate Environment and Install Dependencies
# We use 'source' for Linux/macOS
echo "🛠️  Updating pip and installing dependencies..."
source quantangle_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ Setup Successful!"
    echo "Python version: $(python3 --version)"
    echo "Location: $(which python3)"
    echo "------------------------------------------------"
    echo "💡 To start working:"
    echo "   1. Activate environment: source ./quantangle_venv/bin/activate"
    echo "   2. Run an experiment:    bash ./scripts/n4_m2_L0.95_s-1_rep100.sh"
    echo "   3. Generate plots:       python run_plots.py ./experiments/<folder_name>"
    echo "------------------------------------------------"
else
    echo "❌ Error: Dependency installation failed."
    exit 1
fi