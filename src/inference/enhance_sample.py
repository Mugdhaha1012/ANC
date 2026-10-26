
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.cnn import SpeechEnhancementCNN

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 32000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected 16000 Hz, got {sr}: {path}")

    return audio.mean(axis=1)


def fit_length(audio):
    if len(audio) >= SEGMENT_LENGTH:
        return audio[:SEGMENT_LENGTH]

    return np.pad(audio, (0, SEGMENT_LENGTH - len(audio)))


def calculate_snr(clean, noisy):
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10((signal_power + 1e-12) / noise_power)


def main():
    manifest = ROOT / "data/processed/manifests/test.csv"

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("test.csv is empty")

    # Select the first test example
    sample = rows[0]

    clean_path = ROOT / sample["clean_path"]
    noise_path = ROOT / sample["noise_path"]
    snr_db = float(sample["snr_db"])

    print("Loading test audio...")
    print("SNR:", snr_db, "dB")

    clean = fit_length(load_audio(clean_path))
    noise = fit_length(load_audio(noise_path))

    # Mix noise at the SNR specified in the manifest
    clean_rms = np.sqrt(np.mean(clean ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-12

    noise = noise * (
        clean_rms / (noise_rms * (10 ** (snr_db / 20)))
    )

    noisy = clean + noise

    # Load trained model
    model = SpeechEnhancementCNN().to(DEVICE)

    checkpoint_path = ROOT / "models/best_model.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")
    print("Running AI noise suppression...")

    window = torch.hann_window(
        WIN_LENGTH, device=DEVICE
    )

    noisy_tensor = torch.tensor(
        noisy, dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        noisy_stft = torch.stft(
            noisy_tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True
        )

        noisy_mag = torch.abs(noisy_stft)
        scale = noisy_mag.max().clamp_min(1e-8)
        normalized_mag = noisy_mag / scale

        model_input = torch.log1p(normalized_mag)[None, None, :, :]

        mask = model(model_input)[0, 0]

        enhanced_stft = noisy_stft * mask

        enhanced = torch.istft(
            enhanced_stft,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            length=SEGMENT_LENGTH
        )

    enhanced = enhanced.cpu().numpy()

    # Save all three audio versions at a common gain
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    peak = max(
        np.max(np.abs(clean)),
        np.max(np.abs(noisy)),
        np.max(np.abs(enhanced)),
        1e-8
    )

    gain = min(1.0, 0.99 / peak)

    sf.write(output_dir / "clean_reference.wav", clean * gain, SAMPLE_RATE)
    sf.write(output_dir / "noisy_input.wav", noisy * gain, SAMPLE_RATE)
    sf.write(output_dir / "enhanced_output.wav", enhanced * gain, SAMPLE_RATE)

    input_snr = calculate_snr(clean, noisy)
    output_snr = calculate_snr(clean, enhanced)

    print("\n--- RESULTS ---")
    print(f"Input SNR:    {input_snr:.2f} dB")
    print(f"Output SNR:   {output_snr:.2f} dB")
    print(f"SNR change:   {output_snr - input_snr:.2f} dB")
    print("\nAudio files saved in:", output_dir)
    print("1. clean_reference.wav")
    print("2. noisy_input.wav")
    print("3. enhanced_output.wav")


if __name__ == "__main__":
    main()
