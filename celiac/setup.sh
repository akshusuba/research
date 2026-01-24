#!/bin/bash
# Setup script for celiac-gut-brain-gnn project

set -e

echo "Setting up celiac gut-brain GNN project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other requirements
echo "Installing other dependencies..."
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm

# Install project in development mode
pip install -e .

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"
