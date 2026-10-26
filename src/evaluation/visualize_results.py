
import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pystoi import stoi

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.cnn import SpeechEnhancementCNN
from src.data.audio_dataset import (
    AudioEnhancementDataset,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_snr(clean, estimate):
    noise = clean - estimate
    return 10 * np.log10(
        (np.sum(clean ** 2) + 1e-8) /
        (np.sum(noise ** 2) + 1e-8)
    )


def enhance_audio(model, clean, noise, snr_db):
    noisy = AudioEnhancementDataset.mix_at_snr(
        clean, noise, snr_db
    )

    window = torch.hann_window(WIN_LENGTH)

    noisy_stft = torch.stft(
        noisy,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    noisy_mag = torch.abs(noisy_stft)
    scale = noisy_mag.max().clamp_min(1e-8)

    model_input = torch.log1p(noisy_mag / scale)
    model_input = model_input.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = model(model_input).squeeze().cpu()

    enhanced_stft = mask * noisy_stft

    enhanced = torch.istft(
        enhanced_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=len(clean)
    )

    return noisy, enhanced, noisy_stft, enhanced_stft


# Load dataset
dataset = AudioEnhancementDataset(
    "data/processed/manifests/test.csv"
)

# Load trained model
model = SpeechEnhancementCNN().to(DEVICE)

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ---------------------------------
# 1. Evaluate test samples
# ---------------------------------

input_scores = []
output_scores = []
stoi_scores = []

first_sample = None

for index in range(min(100, len(dataset))):

    rng = random.Random(42 + index)
    row = dataset.rows[index]

    clean = dataset.load_audio(row["clean_path"])
    noise = dataset.load_audio(row["noise_path"])

    clean = dataset.crop_or_pad(clean, rng)
    noise = dataset.crop_or_pad(noise, rng)

    noisy, enhanced, noisy_stft, enhanced_stft = enhance_audio(
        model, clean, noise, float(row["snr_db"])
    )

    clean_np = clean.numpy()
    noisy_np = noisy.numpy()
    enhanced_np = enhanced.numpy()

    input_snr = calculate_snr(clean_np, noisy_np)
    output_snr = calculate_snr(clean_np, enhanced_np)

    input_scores.append(input_snr)
    output_scores.append(output_snr)

    peak = max(
        np.max(np.abs(clean_np)),
        np.max(np.abs(enhanced_np)),
        1e-8
    )

    gain = min(1.0, 0.99 / peak)

    stoi_value = stoi(
        (clean_np * gain).astype(np.float32),
        (enhanced_np * gain).astype(np.float32),
        SAMPLE_RATE,
        extended=False
    )

    stoi_scores.append(stoi_value)

    if index == 0:
        first_sample = (
            clean_np,
            noisy_np,
            enhanced_np,
            noisy_stft,
            enhanced_stft
        )

    if (index + 1) % 10 == 0:
        print(f"Evaluated {index + 1}/100 samples")


input_avg = np.mean(input_scores)
output_avg = np.mean(output_scores)
improvement = output_avg - input_avg
stoi_avg = np.mean(stoi_scores)

print("\n========== RESULTS ==========")
print(f"Average Input SNR: {input_avg:.2f} dB")
print(f"Average Output SNR: {output_avg:.2f} dB")
print(f"SNR Improvement: {improvement:.2f} dB")
print(f"Average STOI: {stoi_avg:.4f}")
print("============================")


# ---------------------------------
# 2. Create judge-ready visualizations
# ---------------------------------

clean, noisy, enhanced, noisy_stft, enhanced_stft = first_sample

time = np.arange(len(clean)) / SAMPLE_RATE

fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    "AI-Based Audio Noise Suppression — Results",
    fontsize=19,
    fontweight="bold"
)

grid = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)

# Waveform comparison
ax1 = fig.add_subplot(grid[0, :])
ax1.plot(time, noisy, label="Noisy Input", alpha=0.65)
ax1.plot(time, enhanced, label="Enhanced Output", alpha=0.85)
ax1.set_title("Waveform Comparison")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Spectrograms
noisy_db = 20 * np.log10(
    np.abs(noisy_stft.numpy()) + 1e-8
)

enhanced_db = 20 * np.log10(
    np.abs(enhanced_stft.numpy()) + 1e-8
)

vmin = min(noisy_db.min(), enhanced_db.min())
vmax = max(noisy_db.max(), enhanced_db.max())

ax2 = fig.add_subplot(grid[1, 0])
im = ax2.imshow(
    noisy_db,
    origin="lower",
    aspect="auto",
    cmap="magma",
    vmin=vmin,
    vmax=vmax
)
ax2.set_title("Noisy Input Spectrogram")
ax2.set_xlabel("Time Frames")
ax2.set_ylabel("Frequency Bins")

ax3 = fig.add_subplot(grid[1, 1])
ax3.imshow(
    enhanced_db,
    origin="lower",
    aspect="auto",
    cmap="magma",
    vmin=vmin,
    vmax=vmax
)
ax3.set_title("Enhanced Output Spectrogram")
ax3.set_xlabel("Time Frames")
ax3.set_ylabel("Frequency Bins")

fig.colorbar(im, ax=[ax2, ax3], label="Magnitude (dB)")

# SNR comparison
ax4 = fig.add_subplot(grid[2, 0])
ax4.bar(
    ["Input SNR", "Output SNR"],
    [input_avg, output_avg]
)
ax4.set_title(f"SNR Comparison (Improvement: {improvement:.2f} dB)")
ax4.set_ylabel("SNR (dB)")
ax4.grid(axis="y", alpha=0.3)

# STOI score
ax5 = fig.add_subplot(grid[2, 1])
ax5.bar(["Average STOI"], [stoi_avg])
ax5.set_ylim(0, 1)
ax5.set_title("Speech Intelligibility (STOI)")
ax5.set_ylabel("Score (0 to 1)")
ax5.grid(axis="y", alpha=0.3)

plt.savefig(
    os.path.join(OUTPUT_DIR, "judge_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved to:")
print("outputs/judge_summary.png")
