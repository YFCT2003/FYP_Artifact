# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：utils.py
@Date：2026/3/13 22:15
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    """Initializes the distributed process group."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    """Cleans up the distributed process group."""
    destroy_process_group()

def setup_logger(name="pMHC_MLM", log_dir="./logs", level=logging.INFO):
    # Only setup logger on the main process
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        # Return a dummy logger for non-main processes
        return logging.getLogger(name)

    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "epitope_mlm.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

logger = setup_logger()

def save_model(model, path):
    # Ensure saving is done on the main process
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return
    try:
        # Unwrap model from DDP if necessary
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(model_state, path)
        logger.info(f"Successfully saved model to: {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {path}, Error: {str(e)}")
        raise

def load_model(model, path, device='cpu'):
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model: {path} (Device: {device})")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {path}, Error: {str(e)}")
        raise

def load_sequences(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'label' in df.columns:
            pos_count = len(df[df['label'] == 1])
            neg_count = len(df[df['label'] == 0])
            logger.info(f"Successfully loaded sequence file: {csv_path}, Positive: {pos_count}, Negative: {neg_count}")
        else:
            logger.info(f"Successfully loaded sequence file: {csv_path}, Total: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load sequence file: {csv_path}, Error: {str(e)}")
        raise

def save_checkpoint(epoch, model, optimizer, scheduler, train_history, val_history, args, path):
    """Saves a complete training checkpoint, handling DDP."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return

    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_history': train_history,
        'val_history': val_history,
        'args': vars(args) if hasattr(args, '__dict__') else args,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Successfully saved checkpoint to: {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {path}, Error: {str(e)}")
        raise

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Loads a complete training checkpoint."""
    try:
        # Load on the CPU first to avoid GPU memory issues, then move to the correct device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # If the model is a DDP model, it might have been saved with a 'module.' prefix.
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            # If not DDP, remove 'module.' prefix if it exists
            if all(key.startswith('module.') for key in state_dict):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        model.to(device)

        result = {
            'model': model,
            'epoch': checkpoint['epoch'],
            'train_history': checkpoint['train_history'],
            'val_history': checkpoint['val_history'],
            'args': checkpoint.get('args', None)
        }

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            result['optimizer'] = optimizer

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            result['scheduler'] = scheduler

        logger.info(f"Successfully loaded checkpoint: {checkpoint_path} (Epoch: {checkpoint['epoch']})")
        return result
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {checkpoint_path}, Error: {str(e)}")
        raise

def calculate_mlm_accuracy(logits, labels):
    """Calculates accuracy for Masked Language Modeling"""
    predictions = torch.argmax(logits, dim=-1)
    # Mask out indices where label is -100 (non-masked tokens)
    mask = (labels != -100)
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / (mask.sum().float() + 1e-8)
    return accuracy.item()

def plot_training_curves(train_history: List[Dict], val_history: List[Dict], save_dir: str, prefix: str = "training"):
    """Plots loss and accuracy separately. Only runs on the main process."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in train_history]

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h['loss'] for h in train_history], label='Train Loss')
    plt.plot(epochs, [h['loss'] for h in val_history], label='Val Loss')
    plt.title(f'{prefix.capitalize()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"))
    plt.close()

    # Plot Accuracy
    if 'accuracy' in train_history[0]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [h['accuracy'] for h in train_history], label='Train Accuracy')
        plt.plot(epochs, [h['accuracy'] for h in val_history], label='Val Accuracy')
        plt.title(f'{prefix.capitalize()} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{prefix}_accuracy.png"))
        plt.close()

class EarlyStopping:
    """Early stops training if validation loss doesn't improve. Handles DDP."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.rank = int(os.environ.get("RANK", 0))

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.rank == 0:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning("Early stopping triggered")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases. Only on main process."""
        if self.rank == 0:
            if self.verbose:
                logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best model...')

            # Unwrap model from DDP
            model_to_save = model.module if isinstance(model, DDP) else model
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(model_to_save.state_dict(), self.path)

        self.val_loss_min = val_loss

def get_device() -> str:
    # This function is less critical in DDP as the device is set by the rank
    return f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu"
