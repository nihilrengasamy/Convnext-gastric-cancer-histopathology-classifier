import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from src.grad_cam import GradCAM, visualize_grad_cam
from src.model import build_model
from src.utils import get_device, load_config


def load_image(image_path, image_size=224):
    """Load and preprocess image."""
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return tf(image).unsqueeze(0), image


def plot_ensemble_predictions(probs, class_names, individual_probs=None, output_path=None):
    """Plot ensemble predictions and individual model predictions."""
    n_models = len(individual_probs) if individual_probs else 0
    n_cols = n_models + 1 if n_models > 0 else 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)
    
    # Ensemble predictions
    colors = ['green' if p > 0.5 else 'red' for p in probs[0]]
    axes[0].bar(class_names, probs[0], color=colors, alpha=0.7)
    axes[0].set_title('Ensemble Prediction')
    axes[0].set_ylabel('Probability')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Individual model predictions
    if individual_probs:
        for idx, model_probs in enumerate(individual_probs):
            colors = ['green' if p > 0.5 else 'red' for p in model_probs[0]]
            axes[idx + 1].bar(class_names, model_probs[0], color=colors, alpha=0.7)
            axes[idx + 1].set_title(f'Model {idx} Prediction')
            axes[idx + 1].set_ylabel('Probability')
            axes[idx + 1].set_ylim([0, 1])
            axes[idx + 1].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved prediction plot to: {output_path}')
    
    plt.show()
    plt.close(fig)


def plot_grad_cam_comparison(image_np, grad_cams, class_names, output_path=None):
    """Plot Grad-CAM visualizations from all models."""
    n_models = len(grad_cams)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
    
    if n_models == 0:
        fig.close()
        return
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = list(axes)
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM from each model
    for idx, cam in enumerate(grad_cams):
        viz = visualize_grad_cam(image_np.copy(), cam)
        axes[idx + 1].imshow(viz)
        axes[idx + 1].set_title(f'Grad-CAM Model {idx}')
        axes[idx + 1].axis('off')
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved Grad-CAM visualization to: {output_path}')
    
    plt.show()
    plt.close(fig)


def main(args):
    device = get_device(args.device)
    print(f'Using device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint['config']
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Build and load model
    ensemble_models = cfg.get('model', {}).get('ensemble_models', ['convnext_tiny', 'convnext_small', 'convnext_base'])
    model = build_model(
        model_name=cfg['model']['name'],
        num_classes=cfg['num_classes'],
        pretrained=False,
        dropout=cfg['model']['dropout'],
        ensemble_models=ensemble_models,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load and preprocess image
    image_tensor, pil_image = load_image(args.image_path, cfg['image_size'])
    image_tensor = image_tensor.to(device)
    
    # Convert image for visualization
    image_np = image_tensor[0].cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    print(f'\nProcessing image: {args.image_path}')
    print(f'Image size: {image_np.shape}')
    print(f'Number of classes: {len(class_names)}')

    with torch.no_grad():
        # Get ensemble prediction
        logits = model(image_tensor)
        ensemble_probs = torch.softmax(logits, dim=1)
        ensemble_pred_idx = ensemble_probs.argmax(dim=1).item()
        ensemble_confidence = ensemble_probs[0, ensemble_pred_idx].item()
        
        # Get individual model predictions if ensemble
        individual_probs = None
        if hasattr(model, 'forward_individual'):
            individual_logits = model.forward_individual(image_tensor)
            individual_probs = [torch.softmax(logits, dim=1).cpu().numpy() for logits in individual_logits]

    print(f'\n' + '='*60)
    print('ENSEMBLE PREDICTION')
    print('='*60)
    print(f'Predicted class: {class_names[ensemble_pred_idx]}')
    print(f'Confidence: {ensemble_confidence:.4f}')
    print(f'\nClass probabilities:')
    for i in range(ensemble_probs.shape[1]):
        prob = ensemble_probs[0, i].item()
        print(f'  {class_names[i]}: {prob:.4f}')

    if individual_probs:
        print(f'\n' + '='*60)
        print('INDIVIDUAL MODEL PREDICTIONS')
        print('='*60)
        for model_idx, model_prob in enumerate(individual_probs):
            pred_idx = model_prob.argmax()
            confidence = model_prob[0, pred_idx]
            print(f'\nModel {model_idx}:')
            print(f'  Predicted class: {class_names[pred_idx]}')
            print(f'  Confidence: {confidence:.4f}')

    # Generate Grad-CAM visualizations
    if args.enable_gradcam or cfg.get('model', {}).get('enable_gradcam', False):
        print(f'\n' + '='*60)
        print('GENERATING GRAD-CAM VISUALIZATIONS')
        print('='*60)
        
        grad_cams = []
        
        try:
            if hasattr(model, 'models'):  # Ensemble
                for sub_model in model.models:
                    try:
                        gc = GradCAM(sub_model, 'backbone.features')
                        cam = gc.generate(image_tensor, class_idx=torch.tensor([[ensemble_pred_idx]], device=device))
                        grad_cams.append(cam)
                        print(f'Generated Grad-CAM for model {len(grad_cams) - 1}')
                    except Exception as e:
                        print(f'Error generating Grad-CAM: {e}')
            else:  # Single model
                try:
                    gc = GradCAM(model, 'backbone.features')
                    cam = gc.generate(image_tensor, class_idx=torch.tensor([[ensemble_pred_idx]], device=device))
                    grad_cams.append(cam)
                    print('Generated Grad-CAM')
                except Exception as e:
                    print(f'Error generating Grad-CAM: {e}')
        except Exception as e:
            print(f'Error with Grad-CAM: {e}')

        if grad_cams:
            plot_grad_cam_comparison(
                image_np, grad_cams, class_names,
                output_path=os.path.join(args.output_dir, 'grad_cam.png') if args.output_dir else None
            )

    # Plot predictions
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_ensemble_predictions(
            ensemble_probs.cpu().numpy(),
            class_names,
            individual_probs=individual_probs,
            output_path=os.path.join(args.output_dir, 'predictions.png')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with Grad-CAM visualization.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth', help='Path to model checkpoint.')
    parser.add_argument('--device', type=str, default='auto', help='auto / mps / cuda / cpu')
    parser.add_argument('--enable_gradcam', action='store_true', help='Enable Grad-CAM visualization.')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Directory to save outputs.')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth', help='Path to model checkpoint.')
    parser.add_argument('--device', type=str, default='auto', help='auto / mps / cuda / cpu')
    args = parser.parse_args()
    main(args)
