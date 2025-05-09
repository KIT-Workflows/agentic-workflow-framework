#!/bin/bash

# Check if repository URL is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <repository_url>"
    echo "Example: $0 https://github.com/username/qe_wf.git"
    exit 1
fi

REPO_URL=$1
REPO_NAME=$(basename "$REPO_URL" .git)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Miniconda installation
check_miniconda() {
    # Check if conda command exists
    if ! command_exists conda; then
        echo "Error: conda is not installed."
        echo "Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi

    # Get conda info
    conda_info=$(conda info | grep -i "base environment")
    
    # Check if it contains "miniconda" (case insensitive)
    if echo "$conda_info" | grep -iq "miniconda"; then
        echo "✓ Miniconda is installed"
        return 0
    else
        echo "Warning: Found conda but it might not be Miniconda."
        echo "This script is tested with Miniconda. Continue anyway? (y/n): "
        read -r response
        if [[ $response =~ ^[Yy] ]]; then
            return 0
        else
            echo "Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
            return 1
        fi
    fi
}

# Check Miniconda installation
if ! check_miniconda; then
    exit 1
fi

# Display installation plan
echo "============================================"
echo "            Installation Plan               "
echo "============================================"
echo "This script will:"
echo "1. Clone repository from: $REPO_URL"
echo ""
echo "2. Create two conda environments:"
echo "   a) qe_wf (main workflow environment)"
echo "      - Python 3.11"
echo "      - FastAPI and web server dependencies"
echo "      - Scientific computing packages (pandas, scikit-learn)"
echo "      - Visualization tools (seaborn)"
echo "      - Other dependencies from requirements.txt"
echo ""
echo "   b) qe (Quantum ESPRESSO environment)"
echo "      - Quantum ESPRESSO from conda-forge"
echo ""
echo "3. Set up configuration:"
echo "   - Copy .env.template to .env (if exists)"
echo "============================================"

# Ask for confirmation
read -p "Do you want to proceed with the installation? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Initialize installation log
INSTALL_LOG=""
append_log() {
    INSTALL_LOG="${INSTALL_LOG}$1\n"
}

# Clone the repository
echo -e "\nStep 1: Cloning repository..."
if git clone "$REPO_URL"; then
    append_log "✓ Repository cloned successfully"
else
    echo "Error: Failed to clone repository"
    exit 1
fi

cd "$REPO_NAME" || { echo "Error: Failed to enter repository directory"; exit 1; }

# Create and activate the main environment
echo -e "\nStep 2a: Creating main environment (qe_wf)..."
if conda create -n qe_wf python=3.11 -y; then
    append_log "✓ Main environment (qe_wf) created successfully"
else
    echo "Error: Failed to create main environment"
    exit 1
fi

# Create QE environment
echo -e "\nStep 2b: Creating QE environment..."
if conda create -n qe -c conda-forge qe -y; then
    append_log "✓ QE environment created successfully"
else
    echo "Error: Failed to create QE environment"
    exit 1
fi

# Function to activate conda environment
activate_conda_env() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        # Linux
        source ~/.bashrc
    fi
    conda activate "$1"
}

# Install requirements in main environment
echo -e "\nStep 2c: Installing requirements in main environment..."
activate_conda_env qe_wf
if pip install -r requirements.txt; then
    append_log "✓ Requirements installed successfully in qe_wf environment"
else
    echo "Error: Failed to install requirements"
    exit 1
fi

# Create .env file from template if it exists
if [ -f .env.template ]; then
    echo -e "\nStep 3: Setting up configuration..."
    if cp .env.template .env; then
        append_log "✓ Created .env file from template"
    else
        append_log "⚠ Failed to create .env file"
    fi
fi

# Display final report
echo -e "\n============================================"
echo "            Installation Report              "
echo "============================================"
echo -e "$INSTALL_LOG"
echo "============================================"
echo -e "\nNext steps:"
echo "1. To activate the environments, use:"
echo "   conda activate qe_wf  # For the main workflow environment"
echo "   conda activate qe     # For the Quantum ESPRESSO environment"
if [ -f .env ]; then
    echo "2. Edit the .env file with your configuration"
fi
echo -e "\nInstallation completed successfully!" 