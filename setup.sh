#!/bin/bash

set -e

echo "==============================================="
echo "ConvNeXt Ensemble + Grad-CAM Setup (macOS)"
echo "==============================================="

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

ARCH=$(uname -m)
echo "Detected architecture: $ARCH"
if [ "$ARCH" = "arm64" ]; then
    echo "✓ Apple Silicon Mac detected"
elif [ "$ARCH" = "x86_64" ]; then
    echo "✓ Intel Mac detected"
else
    echo "Unknown architecture: $ARCH"
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Verifying PyTorch installation..."
python - <<'PYEOF'
import torch
print('✓ PyTorch version:', torch.__version__)
print('✓ MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('✓ CUDA available:', torch.cuda.is_available())
PYEOF

echo ""
echo "==============================================="
echo "✅ Setup complete!"
echo "==============================================="
echo ""
echo "📚 Documentation:"
echo "  - README.md: Overview and quick start"
echo "  - IMPLEMENTATION_GUIDE.md: Detailed technical guide"
echo "  - UPGRADE_SUMMARY.md: What changed"
echo ""
echo "🚀 Quick Start:"
echo "1. Activate: source venv/bin/activate"
echo "2. Train: python train.py --config config.yaml"
echo "3. Evaluate: python test.py --data_dir data/test --checkpoint checkpoints/best.pth --output_dir reports"
echo "4. Infer: python infer.py --image_path sample.jpg --checkpoint checkpoints/best.pth --enable_gradcam"
echo ""
echo "📊 Monitor Training:"
echo "  tensorboard --logdir=runs/convnext_ensemble_classifier/"
echo ""
echo "Evaluate:"
echo "source venv/bin/activate && python test.py --data_dir data/test --checkpoint checkpoints/best.pth"
echo ""
echo "TensorBoard:"
echo "source venv/bin/activate && tensorboard --logdir runs"
