import csv
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, classification_report, roc_curve, auc
)


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device(preferred: str = 'auto'):
    preferred = (preferred or 'auto').lower()
    if preferred == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    if preferred == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        logging.warning('MPS requested but not available. Falling back to CPU.')
        return torch.device('cpu')

    if preferred == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        logging.warning('CUDA requested but not available. Falling back to CPU.')
        return torch.device('cpu')

    return torch.device('cpu')


def setup_logger(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('gastricproject')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def append_metrics_row(csv_path: str, row: dict):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == 'max'
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def compute_metrics(y_true, y_pred, y_proba=None, num_classes=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability predictions (for advanced metrics)
        num_classes: Number of classes
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes) if num_classes else None)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes) if num_classes else None)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes) if num_classes else None)
    
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Advanced metrics if probabilities provided
    if y_proba is not None:
        metrics['log_loss'] = log_loss(y_true, y_proba)
        
        # ROC-AUC for multi-class
        if num_classes and num_classes > 2:
            try:
                # One-vs-Rest ROC-AUC for multi-class
                y_true_bin = np.eye(num_classes)[y_true]
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovo', average='macro')
            except:
                metrics['roc_auc_ovr'] = 0.0
                metrics['roc_auc_ovo'] = 0.0
        else:
            # Binary ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['roc_auc'] = 0.0
    
    return metrics


def compute_cross_validation_metrics(y_true, y_pred_list, y_proba_list, num_classes):
    """
    Compute metrics across cross-validation folds.
    
    Args:
        y_true: True labels
        y_pred_list: List of predictions from each fold
        y_proba_list: List of probabilities from each fold
        num_classes: Number of classes
    
    Returns:
        dict: Mean and std of metrics across folds
    """
    metrics_per_fold = []
    
    for y_pred, y_proba in zip(y_pred_list, y_proba_list):
        fold_metrics = compute_metrics(y_true, y_pred, y_proba, num_classes)
        metrics_per_fold.append(fold_metrics)
    
    cv_metrics = {}
    metric_keys = set()
    for m in metrics_per_fold:
        metric_keys.update(m.keys())
    
    for key in metric_keys:
        values = [m.get(key, 0) for m in metrics_per_fold if isinstance(m.get(key, 0), (int, float))]
        if values:
            cv_metrics[f'{key}_mean'] = np.mean(values)
            cv_metrics[f'{key}_std'] = np.std(values)
    
    return cv_metrics
