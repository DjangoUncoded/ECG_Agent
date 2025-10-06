"""
Complete ECG Analysis Pipeline: Training + Inference

Usage:
1. First time: python ecg_pipeline.py --mode train
2. After training: python ecg_pipeline.py --mode inference --input patient.csv
3. Batch analysis: python ecg_pipeline.py --mode batch --input_dir data/test/
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from torch_ecg.utils.utils_signal import butter_bandpass_filter, normalize
from torch_ecg.models import ECG_CRNN
from torch_ecg.cfg import CFG
from tqdm import tqdm
import argparse
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'classes': ['NSR', 'AF', 'AFL', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'],
    'n_leads': 1,
    'input_len': 5000,
    'fs': 500,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'model_path': 'checkpoints/trained_ecg_model.pth',
    'results_dir': 'results/'
}


# ============================================================
# DATASET CLASS
# ============================================================
class ECGDataset(Dataset):
    """Dataset for ECG CSV files organized by class folders."""

    def __init__(self, data_dir, class_names, fs=500, max_len=5000):
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.fs = fs
        self.max_len = max_len
        self.samples = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for csv_file in class_dir.glob('*.csv'):
                    self.samples.append({
                        'path': csv_file,
                        'label': class_idx,
                        'class_name': class_name
                    })

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ecg = np.loadtxt(sample['path'], delimiter=",")

        if ecg.ndim == 1:
            ecg = ecg[np.newaxis, :]

        # Preprocess
        ecg = butter_bandpass_filter(ecg, lowcut=0.5, highcut=45, fs=self.fs, order=3)
        ecg = normalize(ecg, method="min-max")

        # Pad/truncate
        L = ecg.shape[1]
        if L > self.max_len:
            ecg = ecg[:, :self.max_len]
        elif L < self.max_len:
            pad = np.zeros((ecg.shape[0], self.max_len - L))
            ecg = np.concatenate([ecg, pad], axis=1)

        return torch.tensor(ecg, dtype=torch.float32), torch.tensor(sample['label'], dtype=torch.long)


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(config):
    """Train the ECG classification model."""

    print("\n" + "=" * 70)
    print("TRAINING ECG CLASSIFICATION MODEL")
    print("=" * 70)

    # Check if data exists, create sample data if not
    train_dir = Path('data/train')
    val_dir = Path('data/val')

    if not train_dir.exists() or not val_dir.exists():
        print("\n⚠️  Training data not found. Creating sample dataset...")
        create_sample_dataset(config['classes'][:3])  # Use 3 classes for demo
        config['classes'] = config['classes'][:3]

    # Create datasets
    train_dataset = ECGDataset(train_dir, config['classes'], config['fs'], config['input_len'])
    val_dataset = ECGDataset(val_dir, config['classes'], config['fs'], config['input_len'])

    if len(train_dataset) == 0:
        print("❌ No training data found! Please organize data in data/train/CLASS_NAME/*.csv")
        return None

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Initialize model
    model_config = CFG(classes=config['classes'], input_len=config['input_len'], n_leads=config['n_leads'])
    model = ECG_CRNN(classes=config['classes'], n_leads=config['n_leads'], config=model_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nDevice: {device}")
    print(f"Model: ECG_CRNN with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Classes: {config['classes']}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]")
        for ecg, labels in train_bar:
            ecg, labels = ecg.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(ecg)

            if isinstance(outputs, dict):
                logits = outputs.get('classes', outputs.get('pred', outputs))
            else:
                logits = outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * train_correct / train_total:.2f}%'})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg, labels = ecg.to(device), labels.to(device)

                outputs = model(ecg)
                if isinstance(outputs, dict):
                    logits = outputs.get('classes', outputs.get('pred', outputs))
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'config': config,
                'history': history
            }, config['model_path'])
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'=' * 70}")
    print(f"Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved to: {config['model_path']}")
    print(f"{'=' * 70}\n")

    return model, history


# ============================================================
# INFERENCE & ANALYSIS FUNCTIONS
# ============================================================
def load_trained_model(model_path, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model_config = CFG(classes=config['classes'], input_len=config['input_len'], n_leads=config['n_leads'])
    model = ECG_CRNN(classes=config['classes'], n_leads=config['n_leads'], config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"\n✓ Loaded trained model from {model_path}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"  Classes: {config['classes']}\n")

    return model, config


def detect_r_peaks(ecg_signal, fs=500):
    """Detect R-peaks using signal processing."""
    signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs), prominence=0.5, height=0.3)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs * 1000
    else:
        rr_intervals = np.array([])

    return peaks, rr_intervals


def calculate_hrv_metrics(rr_intervals):
    """Calculate HRV metrics."""
    if len(rr_intervals) < 2:
        return {'mean_rr_ms': 0, 'sdnn_ms': 0, 'rmssd_ms': 0, 'pnn50_percent': 0}

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100

    return {'mean_rr_ms': mean_rr, 'sdnn_ms': sdnn, 'rmssd_ms': rmssd, 'pnn50_percent': pnn50}


def analyze_ecg_file(file_path, model, config, device='cpu'):
    """Perform comprehensive ECG analysis on a single file."""

    # Load ECG
    ecg = np.loadtxt(file_path, delimiter=",")
    if ecg.ndim == 1:
        ecg = ecg[np.newaxis, :]
    ecg_raw = ecg.copy().squeeze()

    # Preprocess for model
    ecg = butter_bandpass_filter(ecg, lowcut=0.5, highcut=45, fs=config['fs'], order=3)
    ecg = normalize(ecg, method="min-max")

    L = ecg.shape[1]
    if L > config['input_len']:
        ecg = ecg[:, :config['input_len']]
        ecg_raw = ecg_raw[:config['input_len']]
    elif L < config['input_len']:
        pad = np.zeros((ecg.shape[0], config['input_len'] - L))
        ecg = np.concatenate([ecg, pad], axis=1)
        ecg_raw = np.concatenate([ecg_raw, np.zeros(config['input_len'] - L)])

    ecg_tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)

    # ML Prediction
    with torch.no_grad():
        outputs = model(ecg_tensor)
        if isinstance(outputs, dict):
            logits = outputs.get('classes', outputs.get('pred', outputs))
        else:
            logits = outputs
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probs)

    # Signal processing analysis
    r_peaks, rr_intervals = detect_r_peaks(ecg_raw, config['fs'])
    hrv = calculate_hrv_metrics(rr_intervals)

    hr_mean = 60 / (np.mean(rr_intervals) / 1000) if len(rr_intervals) > 0 else 0

    # Compile results
    results = {
        'file_path': str(file_path),
        'predicted_class': config['classes'][predicted_class],
        'confidence': float(probs[predicted_class]),
        'heart_rate_bpm': float(hr_mean),
        'num_r_peaks': len(r_peaks),
        'mean_rr_ms': float(hrv['mean_rr_ms']),
        'sdnn_ms': float(hrv['sdnn_ms']),
        'rmssd_ms': float(hrv['rmssd_ms']),
        'pnn50_percent': float(hrv['pnn50_percent']),
    }

    # Add top 3 predictions
    top3 = np.argsort(probs)[-3:][::-1]
    for i, idx in enumerate(top3):
        results[f'top{i + 1}_class'] = config['classes'][idx]
        results[f'top{i + 1}_prob'] = float(probs[idx])

    return results


def batch_analyze(input_dir, model, config, device='cpu'):
    """Analyze all CSV files in a directory."""

    input_path = Path(input_dir)
    csv_files = list(input_path.rglob('*.csv'))

    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {input_dir}")
        return None

    print(f"\n{'=' * 70}")
    print(f"BATCH ANALYSIS: {len(csv_files)} files found")
    print(f"{'=' * 70}\n")

    all_results = []

    for csv_file in tqdm(csv_files, desc="Analyzing ECGs"):
        try:
            results = analyze_ecg_file(csv_file, model, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"⚠️  Error processing {csv_file}: {e}")

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs(config['results_dir'], exist_ok=True)
    output_file = os.path.join(config['results_dir'], 'batch_analysis_results.csv')
    df.to_csv(output_file, index=False)

    print(f"\n✓ Analysis complete! Results saved to {output_file}")
    print(f"\nSummary Statistics:")
    print(df[['predicted_class', 'confidence', 'heart_rate_bpm']].describe())

    return df


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def create_sample_dataset(classes):
    """Create sample dataset for testing."""
    print("Creating sample dataset...")

    for split in ['train', 'val']:
        for cls in classes:
            dir_path = Path(f"data/{split}/{cls}")
            dir_path.mkdir(parents=True, exist_ok=True)

            num_samples = 20 if split == 'train' else 5
            for i in range(num_samples):
                t = np.arange(5000) / 500
                synthetic_ecg = (
                        0.8 * np.sin(2 * np.pi * 1.2 * t) +
                        0.3 * np.sin(2 * np.pi * 3.6 * t) +
                        0.1 * np.random.randn(5000)
                )
                np.savetxt(dir_path / f"sample_{i:03d}.csv", synthetic_ecg, delimiter=",")

    print("✓ Sample dataset created in data/train and data/val")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='ECG Analysis Pipeline')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference', 'batch'],
                        help='Mode: train, inference, or batch')
    parser.add_argument('--input', type=str, help='Input CSV file (for inference) or directory (for batch)')
    parser.add_argument('--model', type=str, default=CONFIG['model_path'], help='Path to model checkpoint')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        model, history = train_model(CONFIG)

    elif args.mode == 'inference':
        if not args.input:
            print("❌ Please specify --input <csv_file>")
            return

        model, config = load_trained_model(args.model, device)
        results = analyze_ecg_file(args.input, model, config, device)

        print(f"\n{'=' * 70}")
        print(f"ANALYSIS RESULTS: {args.input}")
        print(f"{'=' * 70}")
        for key, value in results.items():
            print(f"{key}: {value}")

        # Save individual result
        os.makedirs(CONFIG['results_dir'], exist_ok=True)
        output_file = os.path.join(CONFIG['results_dir'], f"{Path(args.input).stem}_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    elif args.mode == 'batch':
        if not args.input:
            print("❌ Please specify --input_dir <directory>")
            return

        model, config = load_trained_model(args.model, device)
        df = batch_analyze(args.input, model, config, device)


if __name__ == "__main__":
    # If no arguments provided, show usage examples
    import sys

    if len(sys.argv) == 1:
        print("\n" + "=" * 70)
        print("ECG ANALYSIS PIPELINE - Usage Examples")
        print("=" * 70)
        print("\n1️⃣  Train the model:")
        print("   python ecg_pipeline.py --mode train")
        print("\n2️⃣  Analyze a single ECG:")
        print("   python ecg_pipeline.py --mode inference --input patient.csv")
        print("\n3️⃣  Batch analyze multiple ECGs:")
        print("   python ecg_pipeline.py --mode batch --input data/test/")
        print("\n" + "=" * 70 + "\n")
    else:
        main()