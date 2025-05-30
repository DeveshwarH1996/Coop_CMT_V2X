#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import shutil
import argparse
import subprocess
from datetime import datetime
import yaml
import numpy as np

def get_model_checkpoints(model_dir):
    """
    Get all checkpoint files
    """
    epochs = []
    files = os.listdir(model_dir)
    pattern = re.compile(r'net_epoch(\d+)\.pth')
    
    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
    
    return sorted(epochs, reverse=True)

def create_temp_record_dir(base_dir, epoch_num):
    """
    Create temporary record directory and copy necessary files
    """
    # Create temp directory
    temp_dir = os.path.join(base_dir, 'temp_eval', f'epoch_{epoch_num}')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy config file
    shutil.copy2(os.path.join(base_dir, 'config.yaml'), temp_dir)
    
    # Copy and rename model file
    src_model = os.path.join(base_dir, f'net_epoch{epoch_num}.pth')
    dst_model = os.path.join(temp_dir, f'net_epoch{epoch_num}.pth')
    shutil.copy2(src_model, dst_model)
    
    return temp_dir

def extract_ap_values(model_dir, fusion_method, range_val, epoch):
    """
    Extract AP values from log file
    """
    ap_values = None
    
    # Format: eval_{fusion_method}_{range_x}_{range_y}_epoch{epoch}.yaml
    range_x, range_y = range_val.split(',')
    log_file = os.path.join(model_dir, f'eval_{fusion_method}_{range_x}_{range_y}_epoch{epoch}.yaml')
    
    try:
        with open(log_file, 'r') as f:
            data = yaml.safe_load(f)
            if data:
                ap_values = [
                    data.get('ap30', 0.0),
                    data.get('ap_50', 0.0),
                    data.get('ap_70', 0.0)
                ]
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return ap_values

def main():
    parser = argparse.ArgumentParser(description="Evaluate all model checkpoints")
    parser.add_argument('--model_dir', type=str, 
                        default="/home/ecoprt/cooperative_pers/HEAL/opencood/logs/point_pillar_v2xvit_point_transformer_heal_2025_05_14_13_42_12",
                        help='Model directory path')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='Fusion method (late, early, intermediate, no, no_w_uncertainty, single)')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="Detection range in format x,y")
    
    args = parser.parse_args()
    
    # Get all checkpoints
    epochs = get_model_checkpoints(args.model_dir)
    if not epochs:
        print(f"No model checkpoints found in {args.model_dir}")
        return
    
    # Prepare output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.model_dir, f"ap_metrics_{timestamp}.txt")
    
    # Store AP values
    results = []
    
    print(f"Found {len(epochs)} checkpoints. Starting evaluation...")
    
    with open(output_file, 'w') as f:
        f.write(f"Model Checkpoints Evaluation Results - {args.model_dir}\n")
        f.write(f"Fusion Method: {args.fusion_method}\n")
        f.write(f"Detection Range: {args.range}\n")
        f.write(f"Evaluation Time: {timestamp}\n")
        f.write("\n")
        f.write(f"{'Epoch':<10}{'AP@0.3':<10}{'AP@0.5':<10}{'AP@0.7':<10}{'Avg AP':<10}\n")
        f.write("-" * 50 + "\n")
        
        for epoch in epochs[:40]:  # Only evaluate the most recent 40 epochs
            print(f"Evaluating epoch {epoch}...")
            
            # Create temporary directory
            temp_dir = create_temp_record_dir(args.model_dir, epoch)
            
            try:
                # Build inference command
                cmd = [
                    'python', 'opencood/tools/inference.py',
                    '--model_dir', temp_dir,
                    '--fusion_method', args.fusion_method,
                    '--range', args.range
                ]
                
                # Run inference
                subprocess.run(cmd, check=True)
                
                # Extract AP values from log file
                ap_values = extract_ap_values(temp_dir, args.fusion_method, args.range, epoch)
                
                if ap_values:
                    ap30, ap50, ap70 = ap_values
                    avg_ap = sum(ap_values) / 3
                    results.append((epoch, ap30, ap50, ap70, avg_ap))
                    
                    # Write results
                    f.write(f"{epoch:<10}{ap30:<10.2f}{ap50:<10.2f}{ap70:<10.2f}{avg_ap:<10.2f}\n")
                    f.flush()
                else:
                    f.write(f"{epoch:<10}Evaluation Failed\n")
                    f.flush()
            
            finally:
                # Clean up temp directory
                # shutil.rmtree(temp_dir)
                pass
    
    if not results:
        print("No checkpoints were successfully evaluated")
        return
    
    # Find best epochs
    best_ap30 = max(results, key=lambda x: x[1])
    best_ap50 = max(results, key=lambda x: x[2])
    best_ap70 = max(results, key=lambda x: x[3])
    best_avg_ap = max(results, key=lambda x: x[4])
    
    # Add summary
    with open(output_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Evaluation Summary:\n")
        f.write(f"Best AP@0.3: Epoch {best_ap30[0]} (AP: {best_ap30[1]:.2f})\n")
        f.write(f"Best AP@0.5: Epoch {best_ap50[0]} (AP: {best_ap50[2]:.2f})\n")
        f.write(f"Best AP@0.7: Epoch {best_ap70[0]} (AP: {best_ap70[3]:.2f})\n")
        f.write(f"Best Average AP: Epoch {best_avg_ap[0]} (AP: {best_avg_ap[4]:.2f})\n")
    
    print(f"Evaluation complete. Results written to {output_file}")

if __name__ == "__main__":
    main() 