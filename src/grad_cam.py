import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM implementation for generating class activation maps.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Name of the layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                return
        
        raise RuntimeError(f"Target layer '{self.target_layer}' not found in model")
    
    def generate(self, input_image, class_idx=None):
        """
        Generate Grad-CAM map.
        
        Args:
            input_image: Input image tensor (batch_size, 3, H, W)
            class_idx: Class index for which to generate CAM. If None, uses predicted class.
        
        Returns:
            cam: Grad-CAM map (H, W)
        """
        batch_size, _, h, w = input_image.size()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        cam = torch.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()


class IntegratedGradCAM:
    """
    Integrated Gradients with Grad-CAM for better interpretability.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Name of the layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
    
    def generate(self, input_image, class_idx=None, steps=50):
        """
        Generate Integrated Grad-CAM map.
        
        Args:
            input_image: Input image tensor (1, 3, H, W)
            class_idx: Class index for which to generate CAM
            steps: Number of integration steps
        
        Returns:
            cam: Integrated Grad-CAM map (H, W)
        """
        baseline = torch.zeros_like(input_image)
        
        integrated_grads = None
        
        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (input_image - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            output = self.model(interpolated)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1)
            
            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
            output.backward(gradient=one_hot, retain_graph=True)
            
            if integrated_grads is None:
                integrated_grads = interpolated.grad.clone()
            else:
                integrated_grads += interpolated.grad.clone()
        
        integrated_grads /= steps
        integrated_grads = integrated_grads * (input_image - baseline)
        
        # Use integrated gradients as attention map
        cam = integrated_grads[0].abs().mean(dim=0)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.detach().cpu().numpy()


def visualize_grad_cam(image, cam, image_size=(224, 224)):
    """
    Visualize Grad-CAM on the original image.
    
    Args:
        image: Original image (H, W, 3) in numpy format, values [0, 255]
        cam: Grad-CAM map (H, W)
        image_size: Size to resize cam to match image
    
    Returns:
        overlayed_image: Image with Grad-CAM overlay (H, W, 3)
    """
    # Ensure image is correct shape
    if len(image.shape) == 4:
        image = image[0]  # Remove batch dimension if present
    if image.shape[2] != 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on image
    overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return overlayed


def visualize_multiple_cams(image, cams, titles=None):
    """
    Visualize multiple Grad-CAMs side by side.
    
    Args:
        image: Original image (H, W, 3)
        cams: List of CAM arrays
        titles: List of titles for each CAM
    
    Returns:
        combined_image: Concatenated visualization
    """
    if titles is None:
        titles = [f'Model {i}' for i in range(len(cams))]
    
    images_list = [image]
    for cam in cams:
        viz = visualize_grad_cam(image.copy(), cam)
        images_list.append(viz)
    
    # Concatenate horizontally
    combined = np.hstack(images_list)
    
    return combined
