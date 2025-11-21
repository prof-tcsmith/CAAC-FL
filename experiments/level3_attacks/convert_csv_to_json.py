"""
Quick script to convert Level 3 CSV results to JSON format for analysis
"""
import csv
import json
import os
from pathlib import Path

def csv_to_json(csv_path, json_path):
    """Convert metrics CSV to JSON format"""
    metrics = {
        'rounds': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'test_loss': [],
        'attack_type': None,
        'num_byzantine': None,
        'heterogeneity_kl': None,
        'class_imbalance': None
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get metadata from first row
            if metrics['attack_type'] is None and row.get('attack_type'):
                metrics['attack_type'] = row['attack_type']
                metrics['num_byzantine'] = int(row['num_byzantine']) if row.get('num_byzantine') else None
                metrics['heterogeneity_kl'] = float(row['heterogeneity_kl']) if row.get('heterogeneity_kl') else None
                metrics['class_imbalance'] = float(row['class_imbalance']) if row.get('class_imbalance') else None

            # Add round data if present
            if row.get('rounds'):
                metrics['rounds'].append(int(row['rounds']))

            if row.get('test_accuracy'):
                metrics['test_accuracy'].append(float(row['test_accuracy']))

            if row.get('test_loss'):
                metrics['test_loss'].append(float(row['test_loss']))

            if row.get('train_accuracy'):
                metrics['train_accuracy'].append(float(row['train_accuracy']))

            if row.get('train_loss'):
                metrics['train_loss'].append(float(row['train_loss']))

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Converted: {csv_path.name} -> {json_path.name}")

def main():
    results_dir = Path('./results')

    methods = ['fedavg', 'fedmedian', 'krum', 'trimmed_mean']
    attacks = ['no_attack', 'random_noise', 'sign_flipping']

    print("Converting CSV files to JSON format...")
    print("=" * 60)

    for method in methods:
        for attack in attacks:
            csv_file = results_dir / f'level3_{method}_{attack}_metrics.csv'
            json_file = results_dir / f'level3_{method}_{attack}_metrics.json'

            if csv_file.exists():
                try:
                    csv_to_json(csv_file, json_file)
                except Exception as e:
                    print(f"Error converting {csv_file.name}: {e}")
            else:
                print(f"Missing: {csv_file.name}")

    print("=" * 60)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
