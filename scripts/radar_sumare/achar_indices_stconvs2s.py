from pathlib import Path
import argparse
import numpy as np
import torch

from radar_station_memmap_dataset import RadarStationMemmapDataset
from model.stconvs2s import STConvS2S_C


def parse_years(years_arg):
    if "-" in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y.strip()) for y in years_arg.split(",")]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--years", default="2012-2024")
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--top-k", type=int, default=30)
parser.add_argument("--step-scan", type=int, default=1)
parser.add_argument("--t-in", type=int, default=5)
parser.add_argument("--t-out", type=int, default=5)
parser.add_argument("--stride", type=int, default=5)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--width", type=int, default=256)
args = parser.parse_args()

h = args.horizon - 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = RadarStationMemmapDataset(
    radar_root=args.dataset_path,
    years=parse_years(args.years),
    t_in=args.t_in,
    t_out=args.t_out,
    stride=args.stride,
    split="test",
)

model = STConvS2S_C(
    input_size=(1, 3, args.t_in, args.height, args.width),
    num_layers=3,
    hidden_dim=32,
    kernel_size=5,
    device=device,
    dropout_rate=0.0,
    step=5,
    output_channels=1,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["state_dict"]
model.load_state_dict(state)
model.eval()

scores = []

with torch.no_grad():
    for idx in range(0, len(dataset), args.step_scan):
        x, y, m = dataset[idx]

        x_b = x.unsqueeze(0).to(device)
        pred = model(x_b).squeeze(0).cpu().numpy()

        y_np = y.numpy()
        m_np = m.numpy()

        obs_h = y_np[0, h]
        mask_h = m_np[0, h]
        pred_h = pred[0, h]

        valid = mask_h == 1
        if valid.sum() == 0:
            continue

        obs_max = float(obs_h[valid].max())
        obs_mean = float(obs_h[valid].mean())

        pred_clip = np.clip(pred_h, 0, None)
        pred_max_all = float(pred_clip.max())
        pred_mean_all = float(pred_clip.mean())

        pred_max_stations = float(pred_clip[valid].max())
        pred_mean_stations = float(pred_clip[valid].mean())

        scores.append({
            "idx": idx,
            "obs_max": obs_max,
            "obs_mean": obs_mean,
            "pred_max_all": pred_max_all,
            "pred_mean_all": pred_mean_all,
            "pred_max_stations": pred_max_stations,
            "pred_mean_stations": pred_mean_stations,
        })

        if (idx + 1) % 500 == 0:
            print(f"Processado {idx + 1}/{len(dataset)}", flush=True)

print("\n=== Top por maior previsão no mapa inteiro ===")
for s in sorted(scores, key=lambda d: d["pred_max_all"], reverse=True)[:args.top_k]:
    print(
        f"idx={s['idx']} | "
        f"pred_max={s['pred_max_all']:.6f} mm/15min ({s['pred_max_all']*4:.4f} mm/h) | "
        f"obs_max={s['obs_max']:.2f} mm/15min ({s['obs_max']*4:.2f} mm/h)"
    )

print("\n=== Top por maior previsão nas estações ===")
for s in sorted(scores, key=lambda d: d["pred_max_stations"], reverse=True)[:args.top_k]:
    print(
        f"idx={s['idx']} | "
        f"pred_station_max={s['pred_max_stations']:.6f} mm/15min ({s['pred_max_stations']*4:.4f} mm/h) | "
        f"obs_max={s['obs_max']:.2f} mm/15min ({s['obs_max']*4:.2f} mm/h)"
    )

print("\n=== Top por maior chuva observada no horizonte escolhido ===")
for s in sorted(scores, key=lambda d: d["obs_max"], reverse=True)[:args.top_k]:
    print(
        f"idx={s['idx']} | "
        f"obs_max={s['obs_max']:.2f} mm/15min ({s['obs_max']*4:.2f} mm/h) | "
        f"pred_max={s['pred_max_all']:.6f} mm/15min ({s['pred_max_all']*4:.4f} mm/h)"
    )