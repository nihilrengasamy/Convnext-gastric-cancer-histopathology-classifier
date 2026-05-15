import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import build_dataloaders, build_transforms
from src.model import build_model
from src.utils import (
    EarlyStopping,
    append_metrics_row,
    compute_metrics,
    ensure_dir,
    get_device,
    load_config,
    save_config,
    set_seed,
    setup_logger,
)


def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate model on validation/test set."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss /= len(loader.dataset)
    
    # Compute comprehensive metrics
    all_probs = torch.tensor(all_probs).numpy()
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics['loss'] = val_loss
    
    return metrics


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device):
    best_metrics = None
    best_f1 = 0.0
    early_stopping = EarlyStopping(
        patience=cfg['train']['early_stopping_patience'],
        mode='max',
        min_delta=cfg['train'].get('early_stopping_min_delta', 0.0),
    )

    for epoch in range(cfg['train']['epochs']):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg['num_classes'])

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_metrics = val_metrics

        if early_stopping.step(val_metrics['f1_macro']):
            break

    return best_metrics


def summarize_fold_metrics(fold_metrics):
    summary = {}
    metric_keys = set().union(*(m.keys() for m in fold_metrics))

    for key in sorted(metric_keys):
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)

    return summary


def run_cross_validation(cfg):
    set_seed(cfg['seed'])
    ensure_dir(cfg['save_dir'])
    ensure_dir(cfg['log_dir'])

    logger = setup_logger(os.path.join(cfg['log_dir'], 'cv.log'))
    device = get_device(cfg.get('device', 'auto'))
    logger.info('Starting cross-validation with %d folds', cfg['train']['cv_folds'])

    train_tf, eval_tf = build_transforms(cfg['image_size'])
    train_ds = datasets.ImageFolder(cfg['train_dir'], transform=train_tf)
    val_ds = datasets.ImageFolder(cfg['train_dir'], transform=eval_tf)

    labels = [sample[1] for sample in train_ds.samples]
    skf = StratifiedKFold(n_splits=cfg['train']['cv_folds'], shuffle=True, random_state=cfg['seed'])

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_ds)), labels), start=1):
        logger.info('Running fold %d/%d', fold, cfg['train']['cv_folds'])

        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=False,
        )

        ensemble_models = cfg['model'].get('ensemble_models', ['convnext_tiny', 'convnext_small', 'convnext_base'])
        model = build_model(
            model_name=cfg['model']['name'],
            num_classes=cfg['num_classes'],
            pretrained=cfg['model']['pretrained'],
            dropout=cfg['model']['dropout'],
            ensemble_models=ensemble_models,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_name = cfg['optimizer']['name'].lower()
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg['optimizer']['lr'],
                momentum=cfg['optimizer'].get('momentum', 0.9),
                weight_decay=cfg['optimizer']['weight_decay'],
            )
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_name}')

        scheduler_name = cfg['scheduler']['name'].lower()
        if scheduler_name == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
        elif scheduler_name == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg['scheduler'].get('step_size', 5),
                gamma=cfg['scheduler'].get('gamma', 0.1),
            )
        else:
            scheduler = None

        best_fold_metrics = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device)
        if best_fold_metrics is not None:
            fold_metrics.append(best_fold_metrics)
            logger.info('Fold %d best val_f1_macro=%.4f', fold, best_fold_metrics['f1_macro'])
        else:
            logger.warning('Fold %d did not return valid metrics', fold)

    if not fold_metrics:
        logger.error('No valid fold metrics were generated during cross-validation.')
        return

    summary = summarize_fold_metrics(fold_metrics)
    logger.info('Cross-validation summary: %s', summary)

    print('\n' + '='*60)
    print('CROSS-VALIDATION SUMMARY')
    print('='*60)
    for key, value in summary.items():
        print(f'{key}: {value:.4f}')

    summary_path = os.path.join(cfg['log_dir'], 'cv_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Cross-validation summary saved to: {summary_path}')


def train(cfg):
    set_seed(cfg['seed'])
    ensure_dir(cfg['save_dir'])
    ensure_dir(cfg['log_dir'])

    logger = setup_logger(os.path.join(cfg['log_dir'], 'train.log'))
    writer = SummaryWriter(log_dir=cfg['tensorboard_dir'])
    device = get_device(cfg.get('device', 'auto'))
    logger.info('Using device: %s', device)

    save_config(cfg, os.path.join(cfg['save_dir'], 'resolved_config.yaml'))

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        train_dir=cfg['train_dir'],
        val_dir=cfg['val_dir'],
        image_size=cfg['image_size'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )

    logger.info('Class mapping: %s', train_ds.class_to_idx)
    logger.info('Train size: %s | Val size: %s', len(train_ds), len(val_ds))
    logger.info('Number of classes: %s', cfg['num_classes'])

    # Build ensemble model
    ensemble_models = cfg['model'].get('ensemble_models', ['convnext_tiny', 'convnext_small', 'convnext_base'])
    model = build_model(
        model_name=cfg['model']['name'],
        num_classes=cfg['num_classes'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout'],
        ensemble_models=ensemble_models,
    ).to(device)
    
    logger.info('Model: %s with %d ensemble members', cfg['model']['name'], len(ensemble_models))

    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer for all model parameters
    optimizer_name = cfg['optimizer']['name'].lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['optimizer']['lr'],
            momentum=cfg['optimizer'].get('momentum', 0.9),
            weight_decay=cfg['optimizer']['weight_decay'],
        )
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    scheduler_name = cfg['scheduler']['name'].lower()
    if scheduler_name == 'cosineannealinglr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    elif scheduler_name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['scheduler'].get('step_size', 5),
            gamma=cfg['scheduler'].get('gamma', 0.1),
        )
    else:
        scheduler = None

    best_f1 = 0.0
    early_stopping = EarlyStopping(
        patience=cfg['train']['early_stopping_patience'],
        mode='max',
        min_delta=cfg['train'].get('early_stopping_min_delta', 0.0),
    )

    metrics_csv = os.path.join(cfg['log_dir'], 'metrics.csv')

    for epoch in range(cfg['train']['epochs']):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{cfg["train"]["epochs"]}]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        
        val_metrics = evaluate(model, val_loader, criterion, device, cfg['num_classes'])

        # Use Macro F1 as main metric
        val_f1_macro = val_metrics['f1_macro']

        row = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_accuracy': round(train_acc, 6),
            'val_loss': round(val_metrics['loss'], 6),
            'val_accuracy': round(val_metrics['accuracy'], 6),
            'val_precision_macro': round(val_metrics['precision_macro'], 6),
            'val_recall_macro': round(val_metrics['recall_macro'], 6),
            'val_f1_macro': round(val_f1_macro, 6),
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        # Add per-class metrics
        for i in range(cfg['num_classes']):
            row[f'val_precision_class_{i}'] = round(val_metrics.get(f'precision_class_{i}', 0), 6)
            row[f'val_recall_class_{i}'] = round(val_metrics.get(f'recall_class_{i}', 0), 6)
            row[f'val_f1_class_{i}'] = round(val_metrics.get(f'f1_class_{i}', 0), 6)
        
        # Add advanced metrics
        if 'log_loss' in val_metrics:
            row['val_log_loss'] = round(val_metrics['log_loss'], 6)
        if 'roc_auc_ovr' in val_metrics:
            row['val_roc_auc_ovr'] = round(val_metrics['roc_auc_ovr'], 6)
        if 'roc_auc_ovo' in val_metrics:
            row['val_roc_auc_ovo'] = round(val_metrics['roc_auc_ovo'], 6)
        
        append_metrics_row(metrics_csv, row)

        # TensorBoard logging
        writer.add_scalar('train/loss', train_loss, epoch + 1)
        writer.add_scalar('train/accuracy', train_acc, epoch + 1)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch + 1)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch + 1)
        writer.add_scalar('val/precision_macro', val_metrics['precision_macro'], epoch + 1)
        writer.add_scalar('val/recall_macro', val_metrics['recall_macro'], epoch + 1)
        writer.add_scalar('val/f1_macro', val_f1_macro, epoch + 1)
        
        for i in range(cfg['num_classes']):
            writer.add_scalar(f'val/precision_class_{i}', val_metrics.get(f'precision_class_{i}', 0), epoch + 1)
            writer.add_scalar(f'val/recall_class_{i}', val_metrics.get(f'recall_class_{i}', 0), epoch + 1)
            writer.add_scalar(f'val/f1_class_{i}', val_metrics.get(f'f1_class_{i}', 0), epoch + 1)
        
        if 'log_loss' in val_metrics:
            writer.add_scalar('val/log_loss', val_metrics['log_loss'], epoch + 1)
        
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch + 1)

        logger.info(
            'Epoch %s | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1_macro=%.4f',
            epoch + 1, train_loss, train_acc, val_metrics['loss'], val_metrics['accuracy'], val_f1_macro,
        )

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'class_to_idx': train_ds.class_to_idx,
            'config': cfg,
            'val_f1_macro': val_f1_macro,
            'val_accuracy': val_metrics['accuracy'],
        }
        torch.save(checkpoint, os.path.join(cfg['save_dir'], 'last.pth'))

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(checkpoint, os.path.join(cfg['save_dir'], 'best.pth'))
            logger.info('Saved new best checkpoint with val_f1_macro=%.4f', best_f1)

        if early_stopping.step(val_f1_macro):
            logger.info('Early stopping triggered at epoch %s', epoch + 1)
            break

    writer.close()
    logger.info('Training completed. Best validation F1 macro: %.4f', best_f1)
    logger.info('Artifacts saved to: %s', Path(cfg['save_dir']).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ConvNeXt ensemble classifier.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get('train', {}).get('use_cross_validation', False):
        run_cross_validation(config)
    else:
        train(config)
