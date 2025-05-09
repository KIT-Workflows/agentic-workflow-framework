#!/bin/bash

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Miniconda installation
check_miniconda() {
    # Check if conda command exists
    if ! command_exists conda; then
        echo "Error: conda is not installed. Nothing to uninstall."
        return 1
    fi

    # Get conda info
    conda_info=$(conda info | grep -i "base environment")
    
    # Check if it contains "miniconda" (case insensitive)
    if echo "$conda_info" | grep -iq "miniconda"; then
        echo "✓ Found Miniconda installation"
        return 0
    else
        echo "Warning: Found conda but it might not be Miniconda."
        echo "Continue anyway? (y/n): "
        read -r response
        if [[ $response =~ ^[Yy] ]]; then
            return 0
        else
            return 1
        fi
    fi
}

# Check Miniconda installation
if ! check_miniconda; then
    exit 1
fi

# Display uninstallation plan
echo "============================================"
echo "           Uninstallation Plan             "
echo "============================================"
echo "This script will remove:"
echo "1. Conda Environments:"
echo "   - qe_wf (main workflow environment)"
echo "   - qe (Quantum ESPRESSO environment)"
echo ""
echo "2. Local Repository:"
echo "   - Will remove the entire repository directory"
echo "   - Any local changes will be lost"
echo "   - Configuration files (.env) will be removed"
echo "============================================"

# Ask for confirmation
read -p "This will remove all QE workflow components. Are you sure? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Uninstallation cancelled."
    exit 1
fi

# Initialize uninstallation log
UNINSTALL_LOG=""
append_log() {
    UNINSTALL_LOG="${UNINSTALL_LOG}$1\n"
}

# Remove conda environments
echo -e "\nStep 1: Removing conda environments..."

# Remove qe_wf environment
echo "Removing qe_wf environment..."
if conda env remove -n qe_wf -y 2>/dev/null; then
    append_log "✓ Removed qe_wf environment"
else
    append_log "- qe_wf environment not found"
fi

# Remove qe environment
echo "Removing qe environment..."
if conda env remove -n qe -y 2>/dev/null; then
    append_log "✓ Removed qe environment"
else
    append_log "- qe environment not found"
fi

# Get current directory name
CURRENT_DIR=$(basename "$PWD")

# Check if we're in the repository directory
if [[ "$CURRENT_DIR" == "qe_wf" ]]; then
    echo -e "\nStep 2: Removing repository..."
    # Move up one directory since we're about to delete the current directory
    cd ..
    if rm -rf "qe_wf"; then
        append_log "✓ Removed repository directory"
    else
        append_log "⚠ Failed to remove repository directory"
        echo "Warning: Please manually remove the repository directory"
    fi
else
    echo -e "\nNote: Repository directory not found in current location"
    append_log "- Repository directory not found"
fi

# Display final report
echo -e "\n============================================"
echo "           Uninstallation Report            "
echo "============================================"
echo -e "$UNINSTALL_LOG"
echo "============================================"
echo -e "\nUninstallation completed!"
echo "Note: If you installed any global packages or made system-wide changes,"
echo "      you may need to remove those manually." 