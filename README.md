# Gastric Cancer Histopathology Classification

## ConvNeXt Ensemble + Grad‑CAM | PyTorch
A state‑of‑the‑art medical image classification system built using a ConvNeXt ensemble, Grad‑CAM interpretability, and a fully modular PyTorch training/evaluation pipeline.

This project classifies four gastric cancer histopathology tissue types:

-ADI (Adenocarcinoma)
-DEB (Debris)
-LYM (Lymphocytes)
-MUC (Mucin)

The system achieves 96.29% accuracy and 0.9995 ROC‑AUC, with full interpretability and production‑ready evaluation tooling.

## Key Features:
##🔥 Deep Learning Architecture:
->ConvNeXt‑Tiny
->ConvNeXt‑Small
->ConvNeXt‑Base

All combined using learnable ensemble weights for optimal performance.

🧠 Interpretability
Grad‑CAM heatmaps for each model:
->Highlights regions influencing predictions
->Ensures model focuses on relevant histopathological structures
“Interpretability is provided through Grad‑CAM attention maps, which highlight regions contributing most to each prediction.”

## 📊Comprehensive Evaluation
*Accuracy, Precision, Recall, Macro F1
*Per‑class metrics
*Log Loss
*ROC‑AUC (OvR & OvO)
*Confusion Matrix
*Per‑class metric plots
*ROC curves
*Grad‑CAM visualizations

## ⚙️Configurable Training Pipeline
*AdamW optimizer
*CosineAnnealingLR scheduler
*Early stopping
*Cross‑validation support
*YAML‑based configuration

## 📁Project Structure:
project/
├── config.yaml
├── train.py
├── test.py
├── infer.py
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── grad_cam.py
│   └── utils.py
├── checkpoints/
├── logs/
├── reports/
│   ├── test_metrics.json
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── roc_curves.png
│   └── grad_cam/
└── data/
    ├── train/
    ├── val/
    └── test/
    
## 📈 Model Performance:
From the evaluation report:
“Overall accuracy: 96.29%, Macro F1‑Score: 96.19%, ROC‑AUC: 0.99954.”
“Per‑class performance includes ADI F1: 96.57%, DEB F1: 96.77%, LYM F1: 95.88%, MUC F1: 95.53%.”

## Summary:
| Metric | Score |
| --- | --- |
| **Accuracy** | 96.29% |
| **Macro Precision** | 96.12% |
| **Macro Recall** | 96.26% |
| **Macro F1** | 96.19% |
| **Log Loss** | 0.14308 |
| **ROC‑AUC (OvR/OvO)** | 0.99954 / 0.99954 |

## 🏗️ Installation:
git clone <your-repo-url>
cd project
pip install -r requirements.txt

## ⚙️ Configuration:
num_classes: 4
image_size: 224
batch_size: 32

model:
  name: convnext_ensemble
  ensemble_models:
    - convnext_tiny
    - convnext_small
    - convnext_base
  pretrained: true
  dropout: 0.2
  enable_gradcam: true

train:
  epochs: 20
  early_stopping_patience: 8
  
## 🏋️ Training:
python train.py --config config.yaml
Outputs:
*Best/last checkpoints
*TensorBoard logs
*Metrics CSV

## 🧪 Evaluation:
python test.py \
  --data_dir data/test \
  --checkpoint checkpoints/best.pth \
  --output_dir reports
Generates:
->test_metrics.json
->Confusion matrix
->Per‑class metrics plot
->ROC curves
->Grad‑CAM visualizations

## 🔍 Inference + Grad‑CAM:
python infer.py \
  --image_path sample.jpg \
  --checkpoint checkpoints/best.pth \
  --enable_gradcam \
  --output_dir inference_output

## 🧠 Model Architecture (Summary)
“The core model is a ConvNeXt ensemble composed of ConvNeXt‑Tiny, ConvNeXt‑Small, and ConvNeXt‑Base backbones… ensemble output is computed by stacking each model output and applying learnable weights.”

Ensemble Formula
Output
=
∑
𝑖
=
1
3
𝑤
𝑖
⋅
𝑓
𝑖
(
𝑥
)
Where:

𝑓
𝑖
(
𝑥
)
 = output of each ConvNeXt model

𝑤
𝑖
 = learnable ensemble weights

## 📌Future Improvements
->Add Vision Transformer (ViT) backbone
->Mixed precision training
->Class imbalance handling
->ONNX/TorchScript export
->FastAPI deployment
