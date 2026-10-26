
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.audio_dataset import AudioEnhancementDataset
from models.cnn import SpeechEnhancementCNN


# Configuration
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def create_loader(manifest, shuffle):
    dataset = AudioEnhancementDataset(manifest)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    for noisy_log, noisy_mag, clean_mag in loader:

        noisy_log = noisy_log.to(DEVICE)
        noisy_mag = noisy_mag.to(DEVICE)
        clean_mag = clean_mag.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            mask = model(noisy_log)
            enhanced = mask * noisy_mag

            loss = criterion(enhanced, clean_mag)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = noisy_log.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = create_loader(
        "data/processed/manifests/train.csv", True
    )

    val_loader = create_loader(
        "data/processed/manifests/validation.csv", False
    )

    model = SpeechEnhancementCNN().to(DEVICE)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        train_loss = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss = run_epoch(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }

            torch.save(
                checkpoint,
                MODEL_DIR / "best_model.pth"
            )

            print("Saved best model checkpoint.")

        torch.save(
            model.state_dict(),
            MODEL_DIR / "latest_model.pth"
        )

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
