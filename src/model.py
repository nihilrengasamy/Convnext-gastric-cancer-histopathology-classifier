import torch
import torch.nn as nn
from torchvision import models


CONVNEXT_MODELS = {
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
}

CONVNEXT_WEIGHTS = {
    'convnext_tiny': models.ConvNeXt_Tiny_Weights.DEFAULT,
    'convnext_small': models.ConvNeXt_Small_Weights.DEFAULT,
    'convnext_base': models.ConvNeXt_Base_Weights.DEFAULT,
}


class ConvNeXtClassifier(nn.Module):
    """Single ConvNeXt model with classification head."""
    def __init__(self, model_name='convnext_tiny', num_classes=4, pretrained=True, dropout=0.2):
        super().__init__()
        model_name = model_name.lower()
        if model_name not in CONVNEXT_MODELS:
            raise ValueError(f'Unsupported model: {model_name}. Supported: {list(CONVNEXT_MODELS.keys())}')
        
        weights = CONVNEXT_WEIGHTS[model_name] if pretrained else None
        self.backbone = CONVNEXT_MODELS[model_name](weights=weights)
        
        # Replace classification head
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.model_name = model_name
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features from the backbone before classification head."""
        return self.backbone.features(x)


class ConvNeXtEnsemble(nn.Module):
    """Ensemble of ConvNeXt models with weighted averaging."""
    def __init__(self, ensemble_models=['convnext_tiny', 'convnext_small', 'convnext_base'],
                 num_classes=4, pretrained=True, dropout=0.2):
        super().__init__()
        self.models = nn.ModuleList([
            ConvNeXtClassifier(model_name, num_classes, pretrained, dropout)
            for model_name in ensemble_models
        ])
        self.num_classes = num_classes
        self.ensemble_size = len(self.models)
        # Learnable weights for ensemble averaging
        self.ensemble_weights = nn.Parameter(
            torch.ones(self.ensemble_size) / self.ensemble_size
        )
    
    def forward(self, x):
        """Forward pass with ensemble averaging."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs and apply learnable weights
        stacked = torch.stack(outputs, dim=0)  # (ensemble_size, batch_size, num_classes)
        weighted = stacked * self.ensemble_weights.view(-1, 1, 1)
        ensemble_output = weighted.sum(dim=0)  # (batch_size, num_classes)
        
        return ensemble_output
    
    def forward_individual(self, x):
        """Get predictions from each model individually."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return outputs
    
    def get_model(self, idx):
        """Get specific model from ensemble."""
        if idx >= len(self.models):
            raise IndexError(f'Model index {idx} out of range. Ensemble size: {len(self.models)}')
        return self.models[idx]


def build_model(model_name='convnext_ensemble', num_classes=4, pretrained=True, dropout=0.2, 
                ensemble_models=None):
    """Build a ConvNeXt model or ensemble."""
    model_name = model_name.lower()
    
    if model_name == 'convnext_ensemble':
        if ensemble_models is None:
            ensemble_models = ['convnext_tiny', 'convnext_small', 'convnext_base']
        return ConvNeXtEnsemble(
            ensemble_models=ensemble_models,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name in CONVNEXT_MODELS:
        return ConvNeXtClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f'Unsupported model: {model_name}. Supported: {list(CONVNEXT_MODELS.keys())} or convnext_ensemble')
