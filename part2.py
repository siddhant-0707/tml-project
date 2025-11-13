#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part2.py

# This file contains the part2 code for training a model on ImageNette dataset
"""

import sys
import os
import time
import argparse
import urllib.request
import tarfile
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import utils


# ImageNet normalization constants (ImageNette is a subset of ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ImageNette class names
IMAGENETTE_CLASSES = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church',
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
]


def download_imagenette(data_dir='./data', size='160'):
    """
    Download ImageNette dataset if not already present.
    
    Args:
        data_dir: Directory to store dataset
        size: Image size ('160', '320', or 'full')
    
    Returns:
        Path to ImageNette directory
    """
    imagenette_dir = os.path.join(data_dir, 'imagenette2')
    train_dir = os.path.join(imagenette_dir, 'train')
    val_dir = os.path.join(imagenette_dir, 'val')
    
    # Check if dataset already exists
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"ImageNette dataset already exists at {imagenette_dir}")
        return imagenette_dir
    
    print(f"Downloading ImageNette-{size} dataset...")
    url = f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{size}.tgz"
    tgz_path = os.path.join(data_dir, f'imagenette2-{size}.tgz')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, tgz_path)
    print("Download complete. Extracting...")
    
    # Extract
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    
    # Clean up tar file
    os.remove(tgz_path)
    
    # Rename if needed (fastai uses imagenette2, we want imagenette2)
    extracted_dir = os.path.join(data_dir, 'imagenette2')
    if not os.path.exists(extracted_dir):
        # Try alternative name
        for name in os.listdir(data_dir):
            if 'imagenette' in name.lower():
                extracted_dir = os.path.join(data_dir, name)
                break
    
    print(f"ImageNette dataset ready at {extracted_dir}")
    return extracted_dir


def get_imagenette_loaders(batch_size=128, image_size=160, data_dir='./data'):
    """
    Load ImageNette dataset and create train/val/test splits.
    
    Args:
        batch_size: Batch size for data loaders
        image_size: Target image size (160, 320, or 224)
        data_dir: Directory to store/load ImageNette data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Download dataset if needed (use 320px version for better quality when resizing to 224)
    imagenette_dir = download_imagenette(data_dir=data_dir, size='320')
    
    train_dir = os.path.join(imagenette_dir, 'train')
    val_dir = os.path.join(imagenette_dir, 'val')
    
    # Define transforms
    # For training: resize, random crop, horizontal flip, normalize
    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize to 256 first
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # For validation/test: resize, center crop, normalize
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=test_transform)
    
    # For test set, we'll use validation set (ImageNette doesn't have separate test set)
    # In practice, we can split validation further or use val as test
    test_dataset = ImageFolder(root=val_dir, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes: {len(train_dataset.classes)}")
    print(f"  Class names: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader


def normalize_imagenet(x):
    """Normalize images using ImageNet statistics."""
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def get_resnet18_imagenette(num_classes=10):
    """
    Get ResNet-18 model for ImageNette.
    Uses standard ImageNet ResNet-18 architecture (224x224 input).
    """
    # Use standard ResNet-18 (designed for ImageNet/224x224)
    model = torchvision.models.resnet18(weights=None)
    # Adjust final layer for 10 classes (ImageNette has 10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(model, train_loader, optimizer, criterion, device, normalize_fn):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = normalize_fn(inputs.to(device))
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, normalize_fn):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs = normalize_fn(inputs.to(device))
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.1, 
                device="cuda", save_path="part2_model.pt", patience=10):
    """
    Train the model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        lr: Initial learning rate
        device: Device to train on
        save_path: Path to save best model
        patience: Early stopping patience
    
    Returns:
        Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with momentum and learning rate scheduling
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training ResNet-18 on ImageNette")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Max epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, normalize_imagenet
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, normalize_imagenet
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, save_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch+1})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience: {patience})")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
            break
        
        print()
    
    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    print(f"{'='*60}\n")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train model for Part 2')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save-path', type=str, default='part2_model.pt', help='Path to save model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate, do not train')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading ImageNette dataset...")
    train_loader, val_loader, test_loader = get_imagenette_loaders(
        batch_size=args.batch_size,
        image_size=224,  # Standard ImageNet size for ResNet-18
        data_dir=args.data_dir
    )
    
    # Create model
    print("\nCreating ResNet-18 model for ImageNette...")
    model = get_resnet18_imagenette(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.eval_only:
        # Load saved model and evaluate
        if not os.path.exists(args.save_path):
            print(f"Error: Model file {args.save_path} not found!")
            return
        
        print(f"\nLoading model from {args.save_path}...")
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        
        print(f"Model trained for {checkpoint['epoch']+1} epochs")
        print(f"Training accuracy: {checkpoint['train_acc']:.2f}%")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, normalize_imagenet
        )
        
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")
    else:
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=args.save_path,
            patience=args.patience
        )
        
        # Load best model and evaluate on test set
        print("\nEvaluating best model on test set...")
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, normalize_imagenet
        )
        
        print(f"\n{'='*60}")
        print(f"Final Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

