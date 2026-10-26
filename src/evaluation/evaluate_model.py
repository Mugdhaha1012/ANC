import os
import sys
import random
import numpy as np
import torch
from pystoi import stoi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.cnn import SpeechEnhancementCNN
from src.data.audio_dataset import AudioEnhancementDataset, SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_snr(clean, estimate):
    noise = clean - estimate
    return 10 * np.log10(
        (np.sum(clean ** 2) + 1e-8) /
        (np.sum(noise ** 2) + 1e-8)
    )


dataset = AudioEnhancementDataset(
    "data/processed/manifests/test.csv"
)

model = SpeechEnhancementCNN().to(DEVICE)

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

input_scores = []
output_scores = []
stoi_scores = []

window = torch.hann_window(WIN_LENGTH)

for index in range(min(100, len(dataset))):

    rng = random.Random(42 + index)
    row = dataset.rows[index]

    clean = dataset.load_audio(row["clean_path"])
    noise = dataset.load_audio(row["noise_path"])

    clean = dataset.crop_or_pad(clean, rng)
    noise = dataset.crop_or_pad(noise, rng)

    noisy = dataset.mix_at_snr(
        clean, noise, float(row["snr_db"])
    )

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

    noisy_input = torch.log1p(noisy_mag / scale)
    noisy_input = noisy_input.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = model(noisy_input).squeeze().cpu()

    enhanced_stft = mask * noisy_stft

    enhanced = torch.istft(
        enhanced_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=len(clean)
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

    if (index + 1) % 10 == 0:
        print(f"Evaluated {index + 1}/100 samples")


improvements = np.array(output_scores) - np.array(input_scores)

print("\n========== EVALUATION RESULTS ==========")
print(f"Samples evaluated: {len(input_scores)}")
print(f"Average input SNR: {np.mean(input_scores):.2f} dB")
print(f"Average output SNR: {np.mean(output_scores):.2f} dB")
print(f"Average SNR improvement: {np.mean(improvements):.2f} dB")
print(f"Median SNR improvement: {np.median(improvements):.2f} dB")
print(f"Samples improved: {sum(improvements > 0)}/{len(improvements)}")
print(f"Average STOI: {np.mean(stoi_scores):.4f}")
print("========================================")