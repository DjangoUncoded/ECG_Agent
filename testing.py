import torch
import numpy as np
import pandas as pd
from torch_ecg.utils.utils_signal import butter_bandpass_filter, normalize
from torch_ecg.models import ECG_CRNN, ECG_SEQ_LAB_NET
from torch_ecg.cfg import CFG
import matplotlib.pyplot as plt


# ---------------------------
# 1. Load ECG Signal from CSV (Optimized for your format)
# ---------------------------cd
def load_ecg_from_csv(file_path, fs=500, max_len=5000):
    """Load single-column ECG signal from CSV file."""
    try:
        # Load data - single column of voltage values
        ecg_signal = np.loadtxt(file_path, delimiter=",")

        print(f"✓ Raw ECG data loaded: {len(ecg_signal)} samples")
        print(f"  Signal range: [{ecg_signal.min():.3f}, {ecg_signal.max():.3f}]")

        # Ensure it's 1D and convert to 2D: (1, n_samples) for single lead
        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal.reshape(1, -1)  # Shape: (1, n_samples)

        # Apply bandpass filter (0.5-45 Hz for ECG)
        ecg_filtered = butter_bandpass_filter(ecg_signal, lowcut=0.5, highcut=45, fs=fs, order=3)

        # Normalize the signal
        ecg_normalized = normalize(ecg_filtered, method="min-max")

        # Trim or pad to target length
        n_leads, n_samples = ecg_normalized.shape
        if n_samples > max_len:
            ecg_processed = ecg_normalized[:, :max_len]
            print(f"  Trimmed from {n_samples} to {max_len} samples")
        elif n_samples < max_len:
            pad_width = ((0, 0), (0, max_len - n_samples))
            ecg_processed = np.pad(ecg_normalized, pad_width, mode='constant')
            print(f"  Padded from {n_samples} to {max_len} samples")
        else:
            ecg_processed = ecg_normalized
            print(f"  Using all {n_samples} samples")

        # Convert to tensor: (batch_size, n_leads, signal_length)
        ecg_tensor = torch.tensor(ecg_processed, dtype=torch.float32).unsqueeze(0)

        print(f"✓ Final tensor shape: {ecg_tensor.shape}")
        return ecg_tensor

    except Exception as e:
        print(f"❌ Error loading ECG data: {e}")
        # Return synthetic data for testing
        print("Using synthetic data for demonstration")
        return torch.randn(1, 1, max_len)


# ---------------------------
# 2. Initialize Models
# ---------------------------
def setup_models():
    """Initialize ECG analysis models."""

    # Arrhythmia classification model
    class_config = CFG(
        classes=["N", "S", "V", "F", "Q"],  # MIT-BIH classes: Normal, Supraventricular, Ventricular, Fusion, Unknown
        input_len=5000,
        n_leads=1,
    )

    # Segmentation model for R-peak detection
    seg_config = CFG(
        classes=["background", "QRS"],  # Simple segmentation for R-peaks
        input_len=5000,
        n_leads=1,
    )

    try:
        class_model = ECG_CRNN(classes=class_config.classes, n_leads=1, config=class_config)
        seg_model = ECG_SEQ_LAB_NET(classes=seg_config.classes, n_leads=1, config=seg_config)

        # Set to evaluation mode
        class_model.eval()
        seg_model.eval()

        print("✓ Models initialized successfully")
        return class_model, seg_model, class_config

    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return None, None, None


# ---------------------------
# 3. Enhanced Analysis Functions
# ---------------------------
def analyze_ecg_signal(ecg_tensor, class_model, seg_model, class_config):
    """Comprehensive ECG analysis."""
    results = {}

    with torch.no_grad():
        # 1. Arrhythmia Classification
        try:
            class_output = class_model(ecg_tensor)

            if isinstance(class_output, dict):
                logits = class_output.get('logits', class_output.get('pred', class_output))
            else:
                logits = class_output

            # Ensure proper shape
            if logits.dim() > 2:
                logits = logits.mean(dim=-1)

            probs = torch.softmax(logits, dim=-1)
            pred_class_idx = torch.argmax(probs, dim=-1).item()

            results['classification'] = {
                'predicted_class': pred_class_idx,
                'class_name': class_config.classes[pred_class_idx],
                'probabilities': probs.squeeze().cpu().numpy(),
                'confidence': probs.max().item()
            }

        except Exception as e:
            print(f"❌ Classification error: {e}")
            results['classification'] = None

        # 2. R-peak Detection
        try:
            seg_output = seg_model(ecg_tensor)

            if isinstance(seg_output, dict):
                seg_mask = seg_output.get('mask', seg_output.get('pred', seg_output))
            else:
                seg_mask = seg_output

            # For R-peak detection, use the QRS class (index 1)
            if seg_mask.dim() == 3:
                if seg_mask.shape[1] > 1:  # Multiple classes
                    qrs_probs = torch.sigmoid(seg_mask[:, 1, :])  # QRS class
                else:
                    qrs_probs = torch.sigmoid(seg_mask[:, 0, :])  # Single class
            else:
                qrs_probs = torch.sigmoid(seg_mask)

            # Simple peak detection
            rpeaks = (qrs_probs > 0.5).squeeze().cpu().numpy()
            rpeak_indices = np.where(rpeaks)[0]

            results['rpeaks'] = {
                'detected_peaks': len(rpeak_indices),
                'peak_indices': rpeak_indices,
                'qrs_probabilities': qrs_probs.squeeze().cpu().numpy()
            }

        except Exception as e:
            print(f"❌ R-peak detection error: {e}")
            results['rpeaks'] = None

    return results


# ---------------------------
# 4. Visualization
# ---------------------------
def plot_ecg_analysis(ecg_signal, results, fs=500):
    """Plot ECG signal with analysis results."""
    plt.figure(figsize=(15, 10))

    # Time axis
    time = np.arange(len(ecg_signal)) / fs

    # Plot ECG signal
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg_signal, 'b-', linewidth=1, label='ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot R-peak detection if available
    if results.get('rpeaks') and len(results['rpeaks']['peak_indices']) > 0:
        peak_times = results['rpeaks']['peak_indices'] / fs
        peak_values = ecg_signal[results['rpeaks']['peak_indices']]
        plt.plot(peak_times, peak_values, 'ro', markersize=4, label='Detected R-peaks')
        plt.legend()

    # Plot QRS probabilities if available
    if results.get('rpeaks'):
        plt.subplot(2, 1, 2)
        qrs_probs = results['rpeaks']['qrs_probabilities']
        # Ensure same length as ECG signal
        if len(qrs_probs) == len(ecg_signal):
            plt.plot(time, qrs_probs, 'r-', linewidth=1, label='QRS Probability')
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Detection Threshold')
            plt.xlabel('Time (s)')
            plt.ylabel('Probability')
            plt.title('QRS Complex Detection Probabilities')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------
# 5. Main Analysis Pipeline
# ---------------------------
def main():
    print("=" * 70)
    print("ECG Analysis Pipeline - MIT-BIH Format")
    print("=" * 70)

    # Load your ECG data
    file_path = "patient_mitbih.csv"
    ecg_tensor = load_ecg_from_csv(file_path)

    # Initialize models
    class_model, seg_model, class_config = setup_models()

    if class_model is None or seg_model is None:
        print("❌ Failed to initialize models. Using demonstration mode.")
        return

    # Perform analysis
    print("\n" + "=" * 70)
    print("Performing ECG Analysis...")
    print("=" * 70)

    results = analyze_ecg_signal(ecg_tensor, class_model, seg_model, class_config)

    # Display results
    if results['classification']:
        cls_result = results['classification']
        print(f"\n📊 ARRHYTHMIA CLASSIFICATION:")
        print(f"   Predicted Class: {cls_result['class_name']} (Index: {cls_result['predicted_class']})")
        print(f"   Confidence: {cls_result['confidence']:.4f}")
        print(f"   All probabilities:")
        for i, (cls, prob) in enumerate(zip(class_config.classes, cls_result['probabilities'])):
            print(f"     {cls}: {prob:.4f}")

    if results['rpeaks']:
        rpeak_result = results['rpeaks']
        print(f"\n📈 R-PEAK DETECTION:")
        print(f"   Detected {rpeak_result['detected_peaks']} R-peaks")
        if len(rpeak_result['peak_indices']) > 0:
            print(f"   First 10 R-peak locations (samples): {rpeak_result['peak_indices'][:10].tolist()}")

            # Calculate heart rate (approximate)
            if len(rpeak_result['peak_indices']) >= 2:
                rr_intervals = np.diff(rpeak_result['peak_indices'])
                avg_rr = np.mean(rr_intervals)
                heart_rate = 60 * 500 / avg_rr  # Assuming 500 Hz sampling
                print(f"   Estimated Heart Rate: {heart_rate:.1f} BPM")

    # Extract the actual ECG signal for plotting
    ecg_signal_flat = ecg_tensor.squeeze().cpu().numpy()

    # Plot results
    print(f"\n📉 Generating visualization...")
    plot_ecg_analysis(ecg_signal_flat, results)

    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()