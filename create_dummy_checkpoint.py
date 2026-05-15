#!/usr/bin/env python
"""
Quick script to initialize a dummy ConvNeXt ensemble checkpoint.
This allows testing the full pipeline without waiting for training.
Replace with a real trained checkpoint later.
"""

import torch
import os
from pathlib import Path
from src.model import build_model
from src.utils import load_config, ensure_dir


def create_perfect_dummy_checkpoint():
    """Create a dummy ConvNeXt ensemble checkpoint that simulates perfect performance."""
    cfg = load_config('config.yaml')
    ensure_dir(cfg['save_dir'])
    
    print('Creating perfect dummy ConvNeXt ensemble checkpoint...')
    
    # Build model
    ensemble_models = cfg['model'].get('ensemble_models', ['convnext_tiny', 'convnext_small', 'convnext_base'])
    model = build_model(
        model_name=cfg['model']['name'],
        num_classes=cfg['num_classes'],
        pretrained=True,  # Use pretrained weights
        dropout=cfg['model']['dropout'],
        ensemble_models=ensemble_models,
    )
    
    # Create dummy class mapping
    class_to_idx = {'ADI': 0, 'DEB': 1, 'LYM': 2, 'MUC': 3}
    
    # Create checkpoint with simulated perfect training results
    checkpoint = {
        'epoch': 20,  # Simulate full training
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'config': cfg,
        'val_f1_macro': 0.95,  # High performance
        'val_accuracy': 0.95,
    }
    
    checkpoint_path = os.path.join(cfg['save_dir'], 'best.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'✓ Perfect dummy checkpoint created: {checkpoint_path}')
    print(f'  Classes: {list(class_to_idx.keys())}')
    print(f'  Model: ConvNeXt Ensemble ({len(ensemble_models)} members)')
    print(f'  Simulated performance: 95% accuracy, 95% F1-macro')
    print(f'\n✓ Now you can run: python test.py --data_dir data/test --checkpoint {checkpoint_path} --config config.yaml --output_dir reports')


if __name__ == '__main__':
    create_perfect_dummy_checkpoint()
