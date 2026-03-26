"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
from typing import Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    metrics: Dict = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        metrics: Additional metrics to save
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics or {}
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
    
    return save_path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = "cuda"
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with epoch, loss, metrics
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        # Handle PyTorch Lightning checkpoints
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from: {checkpoint_path}")
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Scheduler state loaded")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metrics": checkpoint.get("metrics", {})
    }


def save_model(model: torch.nn.Module, save_path: str) -> str:
    """
    Save model weights only.
    
    Args:
        model: Model to save
        save_path: Path to save weights
        
    Returns:
        Path to saved weights
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
    return save_path


def load_model(model: torch.nn.Module, weight_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load model weights.
    
    Args:
        model: Model to load weights into
        weight_path: Path to weights
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    state_dict = torch.load(weight_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    
    model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded from: {weight_path}")
    
    return model
