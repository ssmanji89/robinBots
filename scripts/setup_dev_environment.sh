#!/bin/bash

# robinBots Development Environment Setup Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Check if running on MacOS M1
if [[ $(uname -m) != "arm64" || $(uname -s) != "Darwin" ]]; then
    print_color $RED "This script is intended for MacOS M1. Exiting."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_color $YELLOW "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    print_color $GREEN "Homebrew is already installed."
fi

# Install Python 3.9
print_color $BLUE "Installing Python 3.9..."
brew install python@3.9

# Set Python 3.9 as the default version
print_color $BLUE "Setting Python 3.9 as default..."
echo 'export PATH="/opt/homebrew/opt/python@3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify Python installation
python3 --version

# Navigate to the project directory
cd ~/Projects/robinBots || exit

# Create and activate virtual environment
print_color $BLUE "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_color $BLUE "Upgrading pip..."
pip install --upgrade pip

# Install project dependencies
print_color $BLUE "Installing project dependencies..."
pip install robin_stocks pandas numpy pytest python-dotenv

# Update requirements.txt
print_color $BLUE "Updating requirements.txt..."
pip freeze > requirements.txt

# Initialize Git repository (if not already initialized)
if [ ! -d .git ]; then
    print_color $BLUE "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Project structure and environment setup"
fi

print_color $GREEN "Development environment setup complete!"
print_color $YELLOW "Remember to activate the virtual environment with 'source venv/bin/activate' when working on this project."