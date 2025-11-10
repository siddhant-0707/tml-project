#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix

import torch

import utils # we need this


######### Prediction Fns #########

"""
## Basic prediction function
"""
@torch.no_grad()
def basic_predict(model, x, device="cuda"):
    x = x.to(device)
    logits = model(x)
    return logits


#### TODO: implement your defense(s) as a new prediction function
#### Make sure it is compatible with the rest of the code in this file:
####    - it needs to take model, x, device
####    - it needs to return logits
#### Note: if your predict function operates on probabilities/labels (instead of logits), that is fine provided you adjust the rest of the code.
#### Put your code here

"""
Defense prediction function combining multiple techniques:
1. Temperature scaling to reduce MIA vulnerability
2. Input randomization and test-time augmentation for adversarial robustness
3. Ensemble predictions for stability
"""
@torch.no_grad()
def defended_predict(model, x, device="cuda", temperature=2.0, num_samples=4, noise_scale=0.01):
    """
    Defended prediction function.
    
    Args:
        model: The target model to protect
        x: Input tensor (batch_size, channels, height, width)
        device: Device to run on
        temperature: Temperature scaling parameter (higher = smoother predictions)
        num_samples: Number of augmented samples for ensemble
        noise_scale: Scale of input randomization noise
    
    Returns:
        Logits (averaged over ensemble, then temperature-scaled)
    """
    model.eval()
    x = x.to(device)
    batch_size = x.shape[0]
    
    # Store predictions for ensemble
    all_logits = []
    
    for _ in range(num_samples):
        # Apply input randomization (small noise + random crops/flips)
        x_aug = x.clone()
        
        # Add small random noise for robustness (data is already normalized)
        if noise_scale > 0:
            noise = torch.randn_like(x_aug) * noise_scale
            x_aug = x_aug + noise
        
        # Random horizontal flip with some probability
        if torch.rand(1).item() > 0.5:
            x_aug = torch.flip(x_aug, dims=[3])
        
        # Small random crop (pad and crop)
        if x_aug.shape[2] == 32:  # CIFAR-10 images
            pad = 2
            x_aug = torch.nn.functional.pad(x_aug, (pad, pad, pad, pad), mode='reflect')
            # Random crop
            h_start = torch.randint(0, 2 * pad + 1, (1,)).item()
            w_start = torch.randint(0, 2 * pad + 1, (1,)).item()
            x_aug = x_aug[:, :, h_start:h_start+32, w_start:w_start+32]
        
        # Get model predictions
        logits = model(x_aug)
        all_logits.append(logits)
    
    # Ensemble: average logits across all augmentations
    ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    
    # Apply temperature scaling to reduce confidence (helps with MIA)
    # This makes predictions smoother and reduces the gap between member/non-member predictions
    scaled_logits = ensemble_logits / temperature
    
    return scaled_logits


######### Membership Inference Attacks (MIAs) #########

"""
## A very simple confidence threshold-based MIA
"""
@torch.no_grad()
def simple_conf_threshold_mia(predict_fn, x, thresh=0.999, device="cuda"):   
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
"""
## A very simple logit threshold-based MIA
"""
@torch.no_grad()
def simple_logits_threshold_mia(predict_fn, x, thresh=9, device="cuda"):   
    pred_y = predict_fn(x, device).cpu().numpy()
    pred_y_max_logit = np.max(pred_y, axis=-1)
    return (pred_y_max_logit > thresh).astype(int)
    
    
"""
## Loss-based MIA - uses prediction loss/entropy as signal
"""
@torch.no_grad()
def loss_based_mia(predict_fn, x, y, device="cuda", loss_type="cross_entropy"):
    """
    Membership inference based on prediction loss.
    Members typically have lower loss than non-members.
    
    Args:
        predict_fn: Prediction function
        x: Input data
        y: True labels
        device: Device to run on
        loss_type: "cross_entropy" or "entropy"
    
    Returns:
        Binary predictions (1 for member, 0 for non-member)
    """
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1)
    
    if loss_type == "cross_entropy":
        # Cross-entropy loss
        y_onehot = torch.zeros_like(pred_y_probas)
        y_onehot.scatter_(1, y.unsqueeze(1).cpu(), 1)
        loss = -torch.sum(y_onehot * torch.log(pred_y_probas + 1e-10), dim=1)
    else:  # entropy
        # Prediction entropy (higher entropy = lower confidence)
        loss = -torch.sum(pred_y_probas * torch.log(pred_y_probas + 1e-10), dim=1)
    
    # Use median threshold (members have lower loss/entropy)
    threshold = torch.median(loss)
    return (loss < threshold).float().numpy().astype(int)


"""
## Entropy-based MIA - uses prediction entropy
"""
@torch.no_grad()
def entropy_based_mia(predict_fn, x, device="cuda", threshold=None):
    """
    Membership inference based on prediction entropy.
    Members typically have lower entropy (higher confidence) than non-members.
    
    Args:
        predict_fn: Prediction function
        x: Input data
        device: Device to run on
        threshold: Entropy threshold (if None, uses median)
    
    Returns:
        Binary predictions (1 for member, 0 for non-member)
    """
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1)
    
    # Calculate entropy
    entropy = -torch.sum(pred_y_probas * torch.log(pred_y_probas + 1e-10), dim=1).numpy()
    
    if threshold is None:
        # Use median threshold
        threshold = np.median(entropy)
    
    # Members have lower entropy (higher confidence)
    return (entropy < threshold).astype(int)


"""
## Modified confidence threshold MIA with adaptive threshold
"""
@torch.no_grad()
def adaptive_conf_threshold_mia(predict_fn, x, device="cuda", percentile=75):
    """
    Membership inference using adaptive confidence threshold.
    Uses percentile-based threshold instead of fixed value.
    
    Args:
        predict_fn: Prediction function
        x: Input data
        device: Device to run on
        percentile: Percentile to use as threshold (default 75th)
    
    Returns:
        Binary predictions (1 for member, 0 for non-member)
    """
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    
    # Adaptive threshold based on percentile
    threshold = np.percentile(pred_y_conf, percentile)
    return (pred_y_conf > threshold).astype(int)


"""
## Likelihood ratio MIA - compares prediction confidence to baseline
"""
@torch.no_grad()
def likelihood_ratio_mia(predict_fn, x, device="cuda", baseline_conf=0.1):
    """
    Membership inference using likelihood ratio.
    Compares prediction confidence to a baseline.
    
    Args:
        predict_fn: Prediction function
        x: Input data
        device: Device to run on
        baseline_conf: Baseline confidence threshold
    
    Returns:
        Binary predictions (1 for member, 0 for non-member)
    """
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    
    # Likelihood ratio: if confidence is much higher than baseline, likely a member
    likelihood_ratio = pred_y_conf / (baseline_conf + 1e-10)
    threshold = 1.5  # Ratio threshold
    return (likelihood_ratio > threshold).astype(int)
  
  
######### Adversarial Examples #########

  
"""
## FGSM (Fast Gradient Sign Method) attack
"""
def fgsm_attack(model, x, y, epsilon=0.03, device="cuda"):
    """
    Fast Gradient Sign Method attack.
    
    Args:
        model: Target model
        x: Input images (normalized)
        y: True labels
        epsilon: Perturbation magnitude
        device: Device to run on
    
    Returns:
        Adversarial examples
    """
    model.eval()
    x = x.to(device).requires_grad_(True)
    y = y.to(device)
    
    # Forward pass
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradient and create adversarial example
    grad = x.grad.data
    x_adv = x + epsilon * grad.sign()
    
    # Clip to valid range (data is normalized, so we clip to reasonable range)
    # For normalized CIFAR-10, values are typically in [-2, 2] range
    x_adv = torch.clamp(x_adv, -3, 3)
    
    return x_adv.detach()


"""
## PGD (Projected Gradient Descent) attack
"""
def pgd_attack(model, x, y, epsilon=0.03, alpha=0.01, num_iter=10, device="cuda", random_start=True):
    """
    Projected Gradient Descent attack (multi-step PGD).
    
    Args:
        model: Target model
        x: Input images (normalized)
        y: True labels
        epsilon: Maximum perturbation magnitude (L-inf norm)
        alpha: Step size per iteration
        num_iter: Number of iterations
        device: Device to run on
        random_start: Whether to start from random perturbation
    
    Returns:
        Adversarial examples
    """
    model.eval()
    x = x.to(device)
    y = y.to(device)
    
    # Initialize adversarial example
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, -3, 3)
    else:
        x_adv = x.clone()
    
    x_adv = x_adv.requires_grad_(True)
    
    for _ in range(num_iter):
        # Forward pass
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial example
        with torch.no_grad():
            grad = x_adv.grad.data
            x_adv = x_adv + alpha * grad.sign()
            
            # Project back to epsilon-ball around original input
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = x + delta
            
            # Clip to valid range
            x_adv = torch.clamp(x_adv, -3, 3)
            x_adv = x_adv.detach().requires_grad_(True)
    
    return x_adv.detach()


"""
## BIM (Basic Iterative Method) attack
"""
def bim_attack(model, x, y, epsilon=0.03, alpha=0.01, num_iter=10, device="cuda"):
    """
    Basic Iterative Method attack (same as PGD but without random start).
    
    Args:
        model: Target model
        x: Input images (normalized)
        y: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size per iteration
        num_iter: Number of iterations
        device: Device to run on
    
    Returns:
        Adversarial examples
    """
    return pgd_attack(model, x, y, epsilon, alpha, num_iter, device, random_start=False)


"""
## Generate and save adversarial examples
"""
def generate_and_save_adversarial(model, loader, attack_fn, attack_name, 
                                   output_file, device="cuda", num_batches=10, **attack_kwargs):
    """
    Generate adversarial examples using specified attack and save to file.
    Data is stored in NCHW format with normalized values (matching advexp0.npz format).
    
    Args:
        model: Target model
        loader: Data loader for benign examples (already normalized)
        attack_fn: Attack function (fgsm_attack, pgd_attack, etc.)
        attack_name: Name of the attack
        output_file: Output file path (.npz)
        device: Device to run on
        num_batches: Number of batches to process
        **attack_kwargs: Additional arguments to pass to attack function
    """
    model.eval()
    all_adv_x = []
    all_benign_x = []
    all_benign_y = []
    
    print(f"Generating {attack_name} adversarial examples...")
    
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        # Generate adversarial examples (data is already normalized from loader)
        x_adv = attack_fn(model, x, y, device=device, **attack_kwargs)
        
        # Convert to numpy (keep in NCHW format, normalized)
        # Detach to remove gradient tracking before converting to numpy
        x_adv_np = x_adv.detach().cpu().numpy().astype(np.float32)
        x_benign_np = x.detach().cpu().numpy().astype(np.float32)
        y_np = y.detach().cpu().numpy().astype(np.int32)
        
        all_adv_x.append(x_adv_np)
        all_benign_x.append(x_benign_np)
        all_benign_y.append(y_np)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1} batches...")
    
    # Concatenate all batches
    adv_x = np.concatenate(all_adv_x, axis=0)
    benign_x = np.concatenate(all_benign_x, axis=0)
    benign_y = np.concatenate(all_benign_y, axis=0)
    
    # Save to file (NCHW format, normalized values)
    np.savez(output_file, adv_x=adv_x, benign_x=benign_x, benign_y=benign_y)
    print(f"Saved {attack_name} adversarial examples to {output_file}")
    print(f"  Shape: adv_x={adv_x.shape}, benign_x={benign_x.shape}, benign_y={benign_y.shape}")
    print(f"  Range: adv_x=[{adv_x.min():.2f}, {adv_x.max():.2f}], benign_x=[{benign_x.min():.2f}, {benign_x.max():.2f}]")
    
    return adv_x, benign_x, benign_y
    
    

def load_and_grab(fp, name, num_batches=4, batch_size=256, shuffle=True):
    loader = utils.make_loader(fp, f"{name}_x", f"{name}_y", batch_size=batch_size, shuffle=shuffle)
    utils.check_loader(loader)
    
    return utils.grab_from_loader(loader, num_batches=num_batches)
    

def load_advex(fp):
    data = np.load(fp)
    return data['adv_x'], data['benign_x'], data['benign_y']
   
######### Main() #########
   
if __name__ == "__main__":

    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Pytorch version: ' + torch.__version__)
    print('------------')

    # deterministic seed for reproducibility
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Device: {device} ---")
    print("-------------------")
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=256, shuffle=False)
    utils.check_loader(train_loader)
    
    # val loader
    val_loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
    utils.check_loader(val_loader)
    
    
    # create the model object   
    model_fp = './target_model.pt'
    assert os.path.exists(model_fp) # model must exist
    
    model, fg = utils.load_model(model_fp, device=device)
    assert fg == "0CCE0F932C863D6648E0", f"Modified model file {model_fp}!"
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and val data
    train_acc = utils.eval_model(model, train_loader, device=device)
    val_acc = utils.eval_model(model, val_loader, device=device)
    print(f"[Raw model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense    
    ### Turn this to True to evaluate your defense (turn it back to False to see the undefended model).
    defense_enabled = True  # Set to True to use defense
    if defense_enabled:
        # Use defended prediction function
        # Parameters can be tuned: temperature (MIA defense), num_samples and noise_scale (adversarial defense)
        predict_fn = lambda x, dev: defended_predict(model, x, device=dev, 
                                                      temperature=2.0, 
                                                      num_samples=4, 
                                                      noise_scale=0.02)
    else:
        # predict_fn points to undefended model
        predict_fn = lambda x, dev: basic_predict(model, x, device=dev)
    
    ### now let's evaluate the model with this prediction function wrapper
    train_acc = utils.eval_wrapper(predict_fn, train_loader, device=device)
    val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)
    
    print(f"[Model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
        
    
    ### evaluating the privacy of the model wrt membership inference
    # load the data
    in_x, in_y = load_and_grab('./data/members.npz', 'members', num_batches=2)
    out_x, out_y = load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)
    
    mia_eval_x = torch.cat([in_x, out_x], 0)
    mia_eval_y = torch.cat([torch.ones_like(in_y), torch.zeros_like(out_y)], 0)
    mia_eval_y = mia_eval_y.cpu().detach().numpy().reshape((-1,1))
    
    assert mia_eval_x.shape[0] == mia_eval_y.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple Conf threshold MIA', simple_conf_threshold_mia))
    mia_attack_fns.append(('Simple Logits threshold MIA', simple_logits_threshold_mia))
    mia_attack_fns.append(('Entropy-based MIA', entropy_based_mia))
    mia_attack_fns.append(('Adaptive Conf threshold MIA', adaptive_conf_threshold_mia))
    mia_attack_fns.append(('Likelihood ratio MIA', likelihood_ratio_mia))
    # Loss-based MIA needs labels, so handle separately
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_x, device=device).reshape((-1,1))       
        assert in_out_preds.shape == mia_eval_y.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_y, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        attack_adv = attack_tpr - attack_fpr
        attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        attack_f1 = tp / (tp + 0.5*(fp + fn)) if (tp + 0.5*(fp + fn)) > 0 else 0.0
        print(f"{attack_str} --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")
    
    # Loss-based MIA (needs labels)
    print('\n------Loss-based MIA (requires labels) ---')
    # Get labels for members and nonmembers
    in_y_labels = load_and_grab('./data/members.npz', 'members', num_batches=2)[1]
    out_y_labels = load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)[1]
    mia_eval_y_labels = torch.cat([in_y_labels, out_y_labels], 0)
    
    in_out_preds_loss = loss_based_mia(predict_fn, mia_eval_x, mia_eval_y_labels, device=device).reshape((-1,1))
    assert in_out_preds_loss.shape == mia_eval_y.shape, 'Invalid attack output format'
    
    cm = confusion_matrix(mia_eval_y, in_out_preds_loss, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    attack_acc = np.trace(cm) / np.sum(np.sum(cm))
    attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    attack_adv = attack_tpr - attack_fpr
    attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_f1 = tp / (tp + 0.5*(fp + fn)) if (tp + 0.5*(fp + fn)) > 0 else 0.0
    print(f"Loss-based MIA (cross_entropy) --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")
    
    in_out_preds_loss_ent = loss_based_mia(predict_fn, mia_eval_x, mia_eval_y_labels, device=device, loss_type="entropy").reshape((-1,1))
    cm = confusion_matrix(mia_eval_y, in_out_preds_loss_ent, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    attack_acc = np.trace(cm) / np.sum(np.sum(cm))
    attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    attack_adv = attack_tpr - attack_fpr
    attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_f1 = tp / (tp + 0.5*(fp + fn)) if (tp + 0.5*(fp + fn)) > 0 else 0.0
    print(f"Loss-based MIA (entropy) --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    
    advexp_fps.append(('Attack0', 'advexp0.npz', '519D7F5E79C3600B366A'))
    
    # Generate new adversarial attacks if files don't exist
    generate_attacks = False  # Set to True to generate new attacks (already generated)
    
    if generate_attacks:
        # Use val loader for generating adversarial examples
        val_loader_adv = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=64, shuffle=False)
        
        # Generate FGSM attack
        if not os.path.exists('advexp_fgsm.npz'):
            generate_and_save_adversarial(model, val_loader_adv, fgsm_attack, 'FGSM', 
                                         'advexp_fgsm.npz', device=device, num_batches=10, epsilon=0.03)
        
        # Generate PGD attack
        if not os.path.exists('advexp_pgd.npz'):
            generate_and_save_adversarial(model, val_loader_adv, pgd_attack, 'PGD', 
                                         'advexp_pgd.npz', device=device, num_batches=10, 
                                         epsilon=0.03, alpha=0.01, num_iter=10, random_start=True)
        
        # Generate BIM attack
        if not os.path.exists('advexp_bim.npz'):
            generate_and_save_adversarial(model, val_loader_adv, bim_attack, 'BIM', 
                                         'advexp_bim.npz', device=device, num_batches=10, 
                                         epsilon=0.03, alpha=0.01, num_iter=10)
    
    # Add generated attacks to evaluation list
    if os.path.exists('advexp_fgsm.npz'):
        advexp_fps.append(('FGSM', 'advexp_fgsm.npz', None))
    if os.path.exists('advexp_pgd.npz'):
        advexp_fps.append(('PGD', 'advexp_pgd.npz', None))
    if os.path.exists('advexp_bim.npz'):
        advexp_fps.append(('BIM', 'advexp_bim.npz', None)) 
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp, attack_hash = tup
        
        assert os.path.exists(attack_fp), f"Attack file {attack_fp} not found."
        _, fg = utils.memv_filehash(attack_fp)
        if attack_hash is not None:
            assert fg == attack_hash, f"Modified attack file {attack_fp}."
        
        # load the attack data
        adv_x, benign_x, benign_y = load_advex(attack_fp)
        benign_y = benign_y.flatten()
        
        # Convert to torch tensor (data is already in NCHW format and normalized)
        benign_x_torch = torch.from_numpy(benign_x).float().to(device)
        adv_x_torch = torch.from_numpy(adv_x).float().to(device)
        
        benign_pred_y = predict_fn(benign_x_torch, device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)
        
        adv_pred_y = predict_fn(adv_x_torch, device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)
        
        hash_str = f" [{fg}]" if attack_hash is not None else ""
        print(f"{attack_str}{hash_str} --- Benign acc: {100*benign_acc:.2f}%; adversarial acc: {100*adv_acc:.2f}%")     
    print('------------\n')

    et = time.time()
    total_sec = et - st
    loading_sec = st_after_model - st
    
    print(f"Elapsed time -- total: {total_sec:.1f} seconds (data & model loading: {loading_sec:.1f} seconds).")
    
    sys.exit(0)
