# Gastric Cancer Histopathology Classification

## ConvNeXt Ensemble + Grad‑CAM | PyTorch
A state‑of‑the‑art medical image classification system built using a ConvNeXt ensemble, Grad‑CAM interpretability, and a fully modular PyTorch training/evaluation pipeline.

This project classifies four gastric cancer histopathology tissue types:<br>

-ADI (Adenocarcinoma)<br>
-DEB (Debris)<br>
-LYM (Lymphocytes)<br>
-MUC (Mucin)<br>

The system achieves 96.29% accuracy and 0.9995 ROC‑AUC, with full interpretability and production‑ready evaluation tooling.

## Key Features:
🔥 Deep Learning Architecture:<br>
->ConvNeXt‑Tiny<br>
->ConvNeXt‑Small<br>
->ConvNeXt‑Base<br>

All combined using learnable ensemble weights for optimal performance.<br>

🧠 Interpretability<br>
Grad‑CAM heatmaps for each model:<br>
->Highlights regions influencing predictions<br>
->Ensures model focuses on relevant histopathological structures<br>
“Interpretability is provided through Grad‑CAM attention maps, which highlight regions contributing most to each prediction.”<br>

## 📊Comprehensive Evaluation
*Accuracy, Precision, Recall, Macro F1<br>
*Per‑class metrics<br>
*Log Loss<br>
*ROC‑AUC (OvR & OvO)<br>
*Confusion Matrix<br>
*Per‑class metric plots<br>
*ROC curves<br>
*Grad‑CAM visualizations<br>

## ⚙️Configurable Training Pipeline
*AdamW optimizer<br>
*CosineAnnealingLR scheduler<br>
*Early stopping<br>
*Cross‑validation support<br>
*YAML‑based configuration<br>

## 📁Project Structure:
project/<br>
├── config.yaml<br>
├── train.py<br>
├── test.py<br>
├── infer.py<br>
├── requirements.txt<br>
├── src/
│   ├── dataset.py<br>
│   ├── model.py<br>
│   ├── grad_cam.py<br>
│   └── utils.py<br>
├── checkpoints/
├── logs/
├── reports/
│   ├── test_metrics.json<br>
│   ├── confusion_matrix.png<br>
│   ├── per_class_metrics.png<br>
│   ├── roc_curves.png<br>
│   └── grad_cam/<br>
└── data/
    ├── train/<br>
    ├── val/<br>
    └── test/<br>
    
## 📈 Model Performance:
From the evaluation report:<br>
“Overall accuracy: 96.29%, Macro F1‑Score: 96.19%, ROC‑AUC: 0.99954.”<br>
“Per‑class performance includes ADI F1: 96.57%, DEB F1: 96.77%, LYM F1: 95.88%, MUC F1: 95.53%.”<br>

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
git clone <your-repo-url><br>
cd project<br>
pip install -r requirements.txt<br>

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
python train.py --config config.yaml<br>
Outputs:
*Best/last <br>
*TensorBoard logs<br>
*Metrics CSV<br>

## 🧪 Evaluation:
python test.py\ <br>
  --data_dir data/test \<br>
  --checkpoint checkpoints/best.pth \<br>
  --output_dir reports<br>
Generates:<br>
->test_metrics.json<br>
->Confusion matrix<br>
->Per‑class metrics plot<br>
->ROC curves<br>
->Grad‑CAM visualizations<br>

## 🔍 Inference + Grad‑CAM
python infer.py \<br>
  --image_path sample.jpg<br>
  --checkpoint checkpoints/best.pth \<br>
  --enable_gradcam \<br>
  --output_dir inference_output

## 🧠 Model Architecture (Summary)
“The core model is a ConvNeXt ensemble composed of ConvNeXt‑Tiny, ConvNeXt‑Small, and ConvNeXt‑Base backbones… ensemble output is computed by stacking each model output and applying learnable weights.”<br>

## 📌Future Improvements
->Add Vision Transformer (ViT) backbone<br>
->Mixed precision training<br>
->Class imbalance handling<br>
->ONNX/TorchScript export<br>
->FastAPI deployment<br>
