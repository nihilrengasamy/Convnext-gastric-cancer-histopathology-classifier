# ConvNeXt Ensemble + Grad-CAM Implementation Guide

## 📋 Overview

This document provides a comprehensive guide to the upgraded ConvNeXt Ensemble + Grad-CAM classification project. The project has been completely refactored from a ResNet binary classifier to a multi-class ensemble deep learning system with advanced interpretability features.

## 🔄 Migration from ResNet Binary to ConvNeXt Ensemble

### Key Changes

#### 1. **Model Architecture**
- **Before**: Single ResNet18/34/50 model
- **After**: Ensemble of three ConvNeXt models (Tiny, Small, Base)
  - Learnable weights for optimal ensemble averaging
  - Individual models can be accessed for Grad-CAM analysis

#### 2. **Data Structure**
- **Before**: Binary classification (2 classes)
- **After**: Multi-class classification (4 classes: ADI, DEB, LYM, MUC)
  - Adapted to multi-class metrics automatically
  - Per-class evaluation supported

#### 3. **Metrics & Evaluation**
- **Before**: Binary precision, recall, F1-score
- **After**: 
  - Macro F1-score (primary metric)
  - Per-class metrics
  - Multi-class ROC-AUC (OvR and OvO)
  - Log Loss for calibration
  - Cross-validation support

#### 4. **Interpretability**
- **New**: Grad-CAM visualization
  - Visual attention maps for each model
  - Integrated Gradients support
  - Per-class activation maps

## 📦 Core Components

### 1. **src/model.py** - Ensemble Architecture

```python
# Single ConvNeXt Classifier
ConvNeXtClassifier(model_name, num_classes, pretrained, dropout)

# Ensemble of ConvNeXt models
ConvNeXtEnsemble(ensemble_models, num_classes, pretrained, dropout)
```

**Key Features**:
- Supports ConvNeXt-Tiny, Small, and Base
- Learnable ensemble weights
- Individual model access for visualization
- Pretrained ImageNet weights

### 2. **src/grad_cam.py** - Interpretability

```python
# Basic Grad-CAM
GradCAM(model, target_layer)
cam = grad_cam.generate(input_image, class_idx)

# Advanced Integrated Gradients
IntegratedGradCAM(model, target_layer)
cam = integrated_grad_cam.generate(input_image, class_idx, steps=50)

# Visualization
visualize_grad_cam(image, cam)
visualize_multiple_cams(image, cams, titles)
```

**Key Features**:
- Layer-specific activation mapping
- Gradient-based attribution
- Multi-model comparison
- Heatmap overlay visualization

### 3. **src/utils.py** - Metrics & Utilities

```python
# Comprehensive metric computation
compute_metrics(y_true, y_pred, y_proba, num_classes)
# Returns: accuracy, precision (macro + per-class), recall, F1, confusion matrix,
#          log_loss, roc_auc (OvR/OvO)

# Cross-validation metrics
compute_cross_validation_metrics(y_true, y_pred_list, y_proba_list, num_classes)
# Returns: mean and std of all metrics across folds
```

**Key Features**:
- Macro-averaged metrics (unweighted by class frequency)
- Per-class metrics for fine-grained analysis
- Advanced metrics (AUC, Log Loss)
- Cross-validation aggregation

### 4. **train.py** - Training with Comprehensive Logging

**Key Changes**:
- Ensemble training with learnable weights
- Macro F1 as primary metric (not accuracy)
- Per-class metrics logged per epoch
- TensorBoard support for all metrics
- Advanced metrics computation

```python
# Main training loop
for epoch in range(cfg['train']['epochs']):
    # Training step
    # Validation step with comprehensive metrics
    # Logging to CSV and TensorBoard
    # Early stopping based on Macro F1
```

### 5. **test.py** - Advanced Evaluation

**Outputs**:
1. **test_metrics.json**: All computed metrics
2. **confusion_matrix.png**: Heatmap visualization
3. **per_class_metrics.png**: Bar charts for each class
4. **roc_curves.png**: ROC curves for all classes
5. **grad_cam/**: Individual Grad-CAM visualizations

### 6. **infer.py** - Single Image Inference

**Features**:
- Ensemble prediction with confidence
- Individual model predictions
- Grad-CAM visualization
- Probability charts
- Output to file or display

## 🎯 Metrics Deep Dive

### Essential Metrics

#### 1. **Accuracy**
Overall proportion of correct predictions:
$$\text{Accuracy} = \frac{\text{# Correct}}{\text{# Total}}$$

#### 2. **Precision (Per-Class)**
Proportion of positive predictions that are correct:
$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$$

#### 3. **Recall (Per-Class)**
Proportion of positive samples that are correctly identified:
$$\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$$

#### 4. **Macro F1-Score** (Primary Metric)
Unweighted average of per-class F1 scores:
$$\text{Macro F1} = \frac{1}{C} \sum_{i=1}^{C} F1_i = \frac{1}{C} \sum_{i=1}^{C} 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

**Why Macro F1?**
- Unbiased by class imbalance
- Treats all classes equally
- Reflects performance on minority classes
- Standard in multi-class medical imaging

### Visualization Metrics

#### 1. **Confusion Matrix**
Shows actual vs predicted for each class. Heatmap colors indicate frequency.

#### 2. **Grad-CAM Maps**
Visual attention regions for each prediction:
- Red: High activation (model focus)
- Blue: Low activation
- Overlaid on original image

#### 3. **Per-Class Metrics Plots**
Bar charts comparing precision, recall, F1 across classes.

### Advanced Metrics

#### 1. **ROC-AUC (Multi-Class)**

**One-vs-Rest (OvR)**:
- Binary classification for each class vs all others
- Average AUC across all one-vs-rest problems
- $$\text{AUC}_{OvR} = \frac{1}{C} \sum_{i=1}^{C} \text{AUC}_{i}$$

**One-vs-One (OvO)**:
- Binary classification for each pair of classes
- Average AUC across all pairwise problems
- More computationally expensive but often more robust

#### 2. **Log Loss (Cross-Entropy)**
Measures probability calibration:
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{p}_{ij})$$

- Penalizes confident but wrong predictions
- Lower is better (min 0, max ∞)
- Useful for probability-based applications

#### 3. **Cross-Validation Metrics**
K-fold cross-validation aggregated results:
- Mean metric across folds
- Standard deviation across folds
- Robustness assessment

## 🚀 Usage Examples

### Training
```bash
python train.py --config config.yaml
```
Outputs: Best/last checkpoints, metrics CSV, TensorBoard logs

### Evaluation
```bash
python test.py \
    --data_dir data/test \
    --checkpoint checkpoints/best.pth \
    --output_dir reports
```
Outputs: All visualizations and metrics JSON

### Inference with Grad-CAM
```bash
python infer.py \
    --image_path sample.jpg \
    --checkpoint checkpoints/best.pth \
    --enable_gradcam \
    --output_dir inference_output
```
Outputs: Predictions, probabilities, Grad-CAM maps

## 🔧 Configuration

### config.yaml Parameters

```yaml
# Model Configuration
model:
  name: convnext_ensemble                    # Type of model
  ensemble_models:                           # Models in ensemble
    - convnext_tiny                          # Fast baseline
    - convnext_small                         # Balanced
    - convnext_base                          # High-capacity
  pretrained: true                           # ImageNet pretrained
  dropout: 0.2                               # Regularization
  enable_gradcam: true                       # Grad-CAM support

# Training Configuration
train:
  epochs: 20                                 # Training epochs
  early_stopping_patience: 8                 # Patience for ES
  early_stopping_min_delta: 0.001            # Min improvement
  cv_folds: 5                                # K-fold CV (if enabled)
  use_cross_validation: false                # Enable CV

# Optimizer Configuration
optimizer:
  name: adamw                                # AdamW optimizer
  lr: 0.0001                                 # Learning rate
  weight_decay: 0.0001                       # L2 regularization

# Scheduler Configuration
scheduler:
  name: cosineannealinglr                    # LR scheduler type
  step_size: 5                               # Step size
  gamma: 0.1                                 # LR decay factor
```

## 📊 Understanding Output Metrics

### Interpret test_metrics.json
```json
{
  "metrics": {
    "accuracy": 0.8750,
    "precision_macro": 0.8733,
    "recall_macro": 0.8750,
    "f1_macro": 0.8741,
    "log_loss": 0.3245,
    "roc_auc_ovr": 0.9567,
    "roc_auc_ovo": 0.9523,
    "precision_class_0": 0.8889,
    "recall_class_0": 0.8889,
    "f1_class_0": 0.8889,
    ...
  },
  "confusion_matrix": [[TP, FP], [FN, TN]]
}
```

### Interpret Confusion Matrix
```
         Predicted
       ADI  DEB  LYM  MUC
       ---- ---- ---- ----
ADI  | 45   2    1    0
DEB  | 1   48    0    0
Actual
LYM  | 1    0   46    2
MUC  | 0    1    1   47
```

Reading:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Row = actual class, Column = predicted class

## 🎨 Visualizations Explained

### confusion_matrix.png
- Heatmap intensity: Prediction frequency
- Diagonal = good, Off-diagonal = errors
- Use to identify which classes are confused

### per_class_metrics.png
- Three subplots: Precision, Recall, F1 for each class
- Identify classes with lower performance
- Compare metrics across classes

### roc_curves.png
- One curve per class (One-vs-Rest)
- AUC score in legend
- Closer to top-left = better
- Diagonal line = random classifier

### grad_cam/gradcam_sample_X.png
- Original image + Grad-CAM from each model
- Red = high activation, Blue = low
- Helps understand model decisions
- Identify if model focuses on relevant regions

## 🔍 Troubleshooting

### Issue: Unbalanced per-class metrics
**Solution**: Macro F1 accounts for this, but check class distribution

### Issue: Grad-CAM not generating
**Solution**: Check target layer name matches model architecture

### Issue: Low ensemble performance
**Solution**: 
- Check individual model performance
- Verify ensemble weights are learning
- Inspect per-class metrics for specific issues

### Issue: Training too slow
**Solution**:
- Reduce batch size
- Use smaller models (ConvNeXt-Tiny only)
- Reduce image size to 196 or 160

## 📚 References

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📝 Version History

### v2.0 (Current)
- ConvNeXt ensemble implementation
- Multi-class support
- Grad-CAM visualization
- Comprehensive metrics
- Cross-validation support

### v1.0 (Previous)
- ResNet binary classification
- Binary metrics
- Basic confusion matrix

---

**Last Updated**: May 5, 2024
**Maintainer**: Your Name
**Questions?** Check GitHub issues or documentation
