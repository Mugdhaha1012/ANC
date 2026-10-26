
import torch
from torch.utils.data import DataLoader

from data.audio_dataset import AudioEnhancementDataset
from models.cnn import SpeechEnhancementCNN


dataset = AudioEnhancementDataset(
    "data/processed/manifests/train.csv"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SpeechEnhancementCNN().to(device)

noisy_log, noisy_mag, clean_mag = next(iter(loader))

noisy_log = noisy_log.to(device)
noisy_mag = noisy_mag.to(device)
clean_mag = clean_mag.to(device)

mask = model(noisy_log)
enhanced = mask * noisy_mag

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
print("Dataset size:", len(dataset))
print("Input shape:", noisy_log.shape)
print("Mask shape:", mask.shape)
print("Enhanced shape:", enhanced.shape)
print("Target shape:", clean_mag.shape)

loss = torch.nn.functional.l1_loss(enhanced, clean_mag)
print("Initial loss:", loss.item())
