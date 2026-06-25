from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from radar_station_memmap_dataset import RadarStationMemmapDataset
from model.stconvs2s import STConvS2S_C


DATASET_PATH = "/home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano"
CHECKPOINT = "/home/noemi/atmoseer/stconvs2s/checkpoint/radar_sumare_2012_2024_15min_256_por_ano_step5_0_20260622-015122.pth.tar"

OUT_DIR = Path("/home/noemi/atmoseer/scripts/radar_sumare/figures/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = RadarStationMemmapDataset(
    radar_root=DATASET_PATH,
    years=list(range(2012, 2025)),
    t_in=5,
    t_out=5,
    stride=5,
    split="test",
)

# Troque esse índice se quiser outro exemplo
idx = 100
x, y, m = dataset[idx]

input_size = (1, 3, 5, 256, 256)

model = STConvS2S_C(
    input_size=input_size,
    num_layers=3,
    hidden_dim=32,
    kernel_size=5,
    device=device,
    dropout_rate=0.0,
    step=5,
    output_channels=1,
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device))

x = x.numpy()
y = y.numpy()
m = m.numpy()
pred = pred.squeeze(0).cpu().numpy()

# última imagem de entrada
radar_img = np.transpose(x[:, -1, :, :], (1, 2, 0))

fig, axes = plt.subplots(5, 3, figsize=(11, 16))

for t in range(5):
    obs = y[0, t]
    mask = m[0, t]
    obs_masked = np.where(mask == 1, obs, np.nan)
    pred_img = pred[0, t]

    axes[t, 0].imshow(radar_img)
    axes[t, 0].set_title("Radar entrada")
    axes[t, 0].axis("off")

    im1 = axes[t, 1].imshow(obs_masked, vmin=0, vmax=12.5)
    axes[t, 1].set_title(f"Observado T+{t+1}")
    axes[t, 1].axis("off")

    im2 = axes[t, 2].imshow(pred_img, vmin=0, vmax=12.5)
    axes[t, 2].set_title(f"STConvS2S T+{t+1}")
    axes[t, 2].axis("off")

fig.colorbar(im2, ax=axes[:, 1:].ravel().tolist(), shrink=0.6, label="mm/15min")
plt.savefig(OUT_DIR / "stconvs2s_mapa_previsao.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figura salva em:", OUT_DIR / "stconvs2s_mapa_previsao.png")