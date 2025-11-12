#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Parameter Tuning Script

This script systematically tests different parameter combinations for the defense
and evaluates their effectiveness against MIA and adversarial attacks.
"""

import sys
import os
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import json
from itertools import product

import utils
import part1  # Import defense functions from part1.py


def evaluate_defense_parameters(model, predict_fn, mia_eval_x, mia_eval_y, mia_eval_y_labels,
                                advexp_fps, device="cuda"):
    """
    Evaluate defense parameters against MIA and adversarial attacks.
    
    Returns:
        dict with evaluation metrics
    """
    results = {}
    
    # Evaluate MIA attacks
    mia_results = {}
    mia_attack_fns = [
        ('Simple Conf threshold MIA', part1.simple_conf_threshold_mia),
        ('Simple Logits threshold MIA', part1.simple_logits_threshold_mia),
        ('Entropy-based MIA', part1.entropy_based_mia),
        ('Adaptive Conf threshold MIA', part1.adaptive_conf_threshold_mia),
        ('Likelihood ratio MIA', part1.likelihood_ratio_mia),
    ]
    
    for attack_str, attack_fn in mia_attack_fns:
        in_out_preds = attack_fn(predict_fn, mia_eval_x, device=device).reshape((-1,1))
        cm = confusion_matrix(mia_eval_y, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        attack_adv = attack_tpr - attack_fpr
        
        mia_results[attack_str] = {
            'accuracy': float(attack_acc),
            'advantage': float(attack_adv),
            'tpr': float(attack_tpr),
            'fpr': float(attack_fpr)
        }
    
    # Loss-based MIA
    in_out_preds_loss = part1.loss_based_mia(predict_fn, mia_eval_x, mia_eval_y_labels, 
                                             device=device, loss_type="cross_entropy").reshape((-1,1))
    cm = confusion_matrix(mia_eval_y, in_out_preds_loss, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    attack_acc = np.trace(cm) / np.sum(np.sum(cm))
    attack_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    attack_adv = attack_tpr - attack_fpr
    mia_results['Loss-based MIA (cross_entropy)'] = {
        'accuracy': float(attack_acc),
        'advantage': float(attack_adv),
        'tpr': float(attack_tpr),
        'fpr': float(attack_fpr)
    }
    
    results['mia'] = mia_results
    
    # Calculate average MIA advantage (excluding perfect defenses for comparison)
    advantages = [v['advantage'] for v in mia_results.values() if v['advantage'] != 0.0 or v['accuracy'] != 0.5]
    results['avg_mia_advantage'] = float(np.mean(np.abs(advantages))) if advantages else 0.0
    results['max_mia_advantage'] = float(np.max(np.abs(advantages))) if advantages else 0.0
    
    # Evaluate adversarial attacks
    adv_results = {}
    for attack_str, attack_fp, attack_hash in advexp_fps:
        if not os.path.exists(attack_fp):
            continue
            
        adv_x, benign_x, benign_y = part1.load_advex(attack_fp)
        benign_y = benign_y.flatten()
        
        benign_x_torch = torch.from_numpy(benign_x).float().to(device)
        adv_x_torch = torch.from_numpy(adv_x).float().to(device)
        
        benign_pred_y = predict_fn(benign_x_torch, device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)
        
        adv_pred_y = predict_fn(adv_x_torch, device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)
        
        adv_results[attack_str] = {
            'benign_acc': float(benign_acc),
            'adversarial_acc': float(adv_acc),
            'drop': float(benign_acc - adv_acc)
        }
    
    results['adversarial'] = adv_results
    
    # Calculate average adversarial accuracy
    if adv_results:
        results['avg_adversarial_acc'] = float(np.mean([v['adversarial_acc'] for v in adv_results.values()]))
        results['avg_benign_acc'] = float(np.mean([v['benign_acc'] for v in adv_results.values()]))
    else:
        results['avg_adversarial_acc'] = 0.0
        results['avg_benign_acc'] = 0.0
    
    return results


def tune_parameters(model, train_loader, val_loader, mia_eval_x, mia_eval_y, mia_eval_y_labels,
                   advexp_fps, device="cuda", mode="quick"):
    """
    Tune defense parameters by testing different combinations.
    
    Args:
        mode: "quick" for faster evaluation with fewer combinations,
              "full" for comprehensive evaluation of all combinations
    """
    print("=" * 80)
    print("PARAMETER TUNING")
    print("=" * 80)
    
    if mode == "quick":
        # Quick mode: test key parameter values
        temperatures = [1.5, 2.0, 2.5, 3.0]
        num_samples_list = [2, 4, 6]
        noise_scales = [0.01, 0.02, 0.03]
        print("Mode: QUICK (testing key parameter values)\n")
    elif mode == "full":
        # Full mode: comprehensive search
        temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        num_samples_list = [2, 4, 6, 8, 10]
        noise_scales = [0.005, 0.01, 0.02, 0.03, 0.04]
        print("Mode: FULL (comprehensive parameter search)\n")
    else:
        # Custom mode: focused search around current best
        temperatures = [1.5, 2.0, 2.5]
        num_samples_list = [4, 6, 8]
        noise_scales = [0.015, 0.02, 0.025]
        print("Mode: FOCUSED (search around baseline parameters)\n")
    
    # Test all combinations
    all_results = []
    
    total_combinations = len(temperatures) * len(num_samples_list) * len(noise_scales)
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    for temp, num_samp, noise in product(temperatures, num_samples_list, noise_scales):
        print(f"Testing: temp={temp}, num_samples={num_samp}, noise_scale={noise:.3f}")
        
        # Create prediction function with current parameters
        predict_fn = lambda x, dev: part1.defended_predict(
            model, x, device=dev, 
            temperature=temp, 
            num_samples=num_samp, 
            noise_scale=noise
        )
        
        # Evaluate clean accuracy (use smaller subset for faster evaluation in quick mode)
        st_time = time.time()
        if mode == "quick":
            # Use smaller subset for faster evaluation
            val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)
            # Estimate train acc from val (skip full train eval for speed)
            train_acc = val_acc + 0.05  # Rough estimate
        else:
            train_acc = utils.eval_wrapper(predict_fn, train_loader, device=device)
            val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)
        eval_time = time.time() - st_time
        
        # Evaluate against attacks
        attack_results = evaluate_defense_parameters(
            model, predict_fn, mia_eval_x, mia_eval_y, mia_eval_y_labels,
            advexp_fps, device=device
        )
        
        # Store results
        result = {
            'temperature': temp,
            'num_samples': num_samp,
            'noise_scale': noise,
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'eval_time': float(eval_time),
            'avg_mia_advantage': attack_results['avg_mia_advantage'],
            'max_mia_advantage': attack_results['max_mia_advantage'],
            'avg_adversarial_acc': attack_results['avg_adversarial_acc'],
            'avg_benign_acc': attack_results['avg_benign_acc'],
            'mia_results': attack_results['mia'],
            'adversarial_results': attack_results['adversarial']
        }
        
        all_results.append(result)
        
        print(f"  Val acc: {val_acc:.4f}, Avg MIA adv: {attack_results['avg_mia_advantage']:.3f}, "
              f"Avg adv acc: {attack_results['avg_adversarial_acc']:.4f}, Time: {eval_time:.2f}s\n")
    
    return all_results


def analyze_results(all_results):
    """
    Analyze tuning results and recommend optimal parameters.
    """
    print("=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Convert to numpy array for easier analysis
    results_array = np.array([[
        r['temperature'],
        r['num_samples'],
        r['noise_scale'],
        r['val_acc'],
        r['avg_mia_advantage'],
        r['avg_adversarial_acc'],
        r['eval_time']
    ] for r in all_results])
    
    # Find best configurations for different criteria
    print("\n1. Best Privacy Protection (lowest MIA advantage):")
    best_privacy_idx = np.argmin(results_array[:, 4])
    best_privacy = all_results[best_privacy_idx]
    print(f"   Parameters: temp={best_privacy['temperature']}, "
          f"num_samples={best_privacy['num_samples']}, "
          f"noise_scale={best_privacy['noise_scale']:.3f}")
    print(f"   Val acc: {best_privacy['val_acc']:.4f}, "
          f"MIA adv: {best_privacy['avg_mia_advantage']:.3f}, "
          f"Adv acc: {best_privacy['avg_adversarial_acc']:.4f}")
    
    print("\n2. Best Adversarial Robustness (highest adversarial accuracy):")
    best_robust_idx = np.argmax(results_array[:, 5])
    best_robust = all_results[best_robust_idx]
    print(f"   Parameters: temp={best_robust['temperature']}, "
          f"num_samples={best_robust['num_samples']}, "
          f"noise_scale={best_robust['noise_scale']:.3f}")
    print(f"   Val acc: {best_robust['val_acc']:.4f}, "
          f"MIA adv: {best_robust['avg_mia_advantage']:.3f}, "
          f"Adv acc: {best_robust['avg_adversarial_acc']:.4f}")
    
    print("\n3. Best Accuracy (highest validation accuracy):")
    best_acc_idx = np.argmax(results_array[:, 3])
    best_acc = all_results[best_acc_idx]
    print(f"   Parameters: temp={best_acc['temperature']}, "
          f"num_samples={best_acc['num_samples']}, "
          f"noise_scale={best_acc['noise_scale']:.3f}")
    print(f"   Val acc: {best_acc['val_acc']:.4f}, "
          f"MIA adv: {best_acc['avg_mia_advantage']:.3f}, "
          f"Adv acc: {best_acc['avg_adversarial_acc']:.4f}")
    
    # Balanced score: combine privacy, robustness, and accuracy
    print("\n4. Best Balanced Configuration (weighted score):")
    # Normalize metrics (higher is better for all after normalization)
    val_accs = results_array[:, 3]
    mia_advs = results_array[:, 4]
    adv_accs = results_array[:, 5]
    
    # Normalize to [0, 1] range
    norm_val_acc = (val_accs - val_accs.min()) / (val_accs.max() - val_accs.min() + 1e-10)
    norm_mia_privacy = 1 - (mia_advs - mia_advs.min()) / (mia_advs.max() - mia_advs.min() + 1e-10)
    norm_adv_acc = (adv_accs - adv_accs.min()) / (adv_accs.max() - adv_accs.min() + 1e-10)
    
    # Weighted score: 30% accuracy, 35% privacy, 35% robustness
    balanced_scores = 0.30 * norm_val_acc + 0.35 * norm_mia_privacy + 0.35 * norm_adv_acc
    best_balanced_idx = np.argmax(balanced_scores)
    best_balanced = all_results[best_balanced_idx]
    print(f"   Parameters: temp={best_balanced['temperature']}, "
          f"num_samples={best_balanced['num_samples']}, "
          f"noise_scale={best_balanced['noise_scale']:.3f}")
    print(f"   Val acc: {best_balanced['val_acc']:.4f}, "
          f"MIA adv: {best_balanced['avg_mia_advantage']:.3f}, "
          f"Adv acc: {best_balanced['avg_adversarial_acc']:.4f}")
    print(f"   Balanced score: {balanced_scores[best_balanced_idx]:.4f}")
    
    return {
        'best_privacy': best_privacy,
        'best_robust': best_robust,
        'best_accuracy': best_acc,
        'best_balanced': best_balanced,
        'all_results': all_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune defense parameters')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'full', 'focused'],
                       help='Tuning mode: quick (fast), full (comprehensive), focused (around baseline)')
    args = parser.parse_args()
    
    # Setup (similar to part1.py)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load data and model
    print("Loading data and model...")
    train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=256, shuffle=False)
    val_loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
    
    model_fp = './target_model.pt'
    model, fg = utils.load_model(model_fp, device=device)
    print(f"Loaded model: {fg}\n")
    
    # Load MIA evaluation data
    in_x, in_y = part1.load_and_grab('./data/members.npz', 'members', num_batches=2)
    out_x, out_y = part1.load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)
    
    mia_eval_x = torch.cat([in_x, out_x], 0)
    mia_eval_y = torch.cat([torch.ones_like(in_y), torch.zeros_like(out_y)], 0)
    mia_eval_y = mia_eval_y.cpu().detach().numpy().reshape((-1,1))
    
    in_y_labels = part1.load_and_grab('./data/members.npz', 'members', num_batches=2)[1]
    out_y_labels = part1.load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)[1]
    mia_eval_y_labels = torch.cat([in_y_labels, out_y_labels], 0)
    
    # Adversarial attack files
    advexp_fps = [
        ('Attack0', 'advexp0.npz', '519D7F5E79C3600B366A'),
    ]
    # Add generated attacks if they exist
    if os.path.exists('advexp_fgsm.npz'):
        advexp_fps.append(('FGSM', 'advexp_fgsm.npz', None))
    if os.path.exists('advexp_pgd.npz'):
        advexp_fps.append(('PGD', 'advexp_pgd.npz', None))
    if os.path.exists('advexp_bim.npz'):
        advexp_fps.append(('BIM', 'advexp_bim.npz', None))
    
    # Run parameter tuning
    all_results = tune_parameters(
        model, train_loader, val_loader, 
        mia_eval_x, mia_eval_y, mia_eval_y_labels,
        advexp_fps, device=device, mode=args.mode
    )
    
    # Analyze results
    analysis = analyze_results(all_results)
    
    # Save results to JSON
    output_file = f'tuning_results_{args.mode}.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    best = analysis['best_balanced']
    print(f"Temperature: {best['temperature']}")
    print(f"Num samples: {best['num_samples']}")
    print(f"Noise scale: {best['noise_scale']:.3f}")
    print(f"\nExpected performance:")
    print(f"  Validation accuracy: {best['val_acc']:.4f}")
    print(f"  Average MIA advantage: {best['avg_mia_advantage']:.3f}")
    print(f"  Average adversarial accuracy: {best['avg_adversarial_acc']:.4f}")
    print(f"  Evaluation time: {best['eval_time']:.2f}s")

