#!/usr/bin/env bash

set -e  # stop on error

echo "🚀 Starting environment setup..."

# -------------------------------
# 1. System Update + Essentials
# -------------------------------
echo "📦 Installing system dependencies..."
apt update && apt install -y \
    build-essential \
    curl \
    git \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# -------------------------------
# 2. Install uv (fast pip)
# -------------------------------
echo "⚡ Installing uv..."
curl -Ls https://astral.sh/uv/install.sh | bash

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# -------------------------------
# 3. Create Virtual Environment
# -------------------------------
echo "🐍 Creating virtual environment..."
python3.10 -m venv myenv

source myenv/bin/activate

# -------------------------------
# 4. Fix pip inside venv
# -------------------------------
echo "🔧 Fixing pip..."
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

# -------------------------------
# 5. Install PyTorch (GPU version compatible with CUDA 11.8)
# -------------------------------


echo "🔥 Installing PyTorch (GPU - CUDA 11.8 compatible)..."

uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# -------------------------------
# 6. Install compatible dependencies
# -------------------------------
echo "📚 Installing core dependencies..."


# uv pip install --r requirements.txt || echo "⚠️ Some packages failed, continuing..."

uv pip install \
    "transformers>=4.30" \
    "diffusers>=0.20" \
    "huggingface_hub<0.20" \
    "accelerate" \
    "safetensors" \
    "pillow" \
    "matplotlib" \
    "scikit-learn" \
    "pandas" \
    "numpy" \
    "datasets" \
    "streamlit" \
    "python-dotenv"

# -------------------------------
# 7. Install DAAM (from GitHub)
# -------------------------------
echo "🧠 Installing DAAM..."
uv pip install git+https://github.com/castorini/daam.git

# -------------------------------
# 8. Install requirements.txt (safe mode)
# -------------------------------
if [ -f "requirements.txt" ]; then
    echo "📄 Installing requirements.txt..."
    uv pip install -r requirements.txt || echo "⚠️ Some packages failed, continuing..."
fi

# -------------------------------
# 9. Hugging Face login reminder
# -------------------------------
echo ""
echo "🔐 IMPORTANT: Login to HuggingFace"
echo "Run: huggingface-cli login"
echo ""

# -------------------------------
# 10. Done
# -------------------------------
echo "✅ Setup complete!"
echo "👉 Activate env: source myenv/bin/activate"