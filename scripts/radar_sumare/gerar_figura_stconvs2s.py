from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from radar_station_memmap_dataset import RadarStationMemmapDataset
from model.stconvs2s import STConvS2S_C


def parse_years(years_arg):
    if "-" in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y.strip()) for y in years_arg.split(",")]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--years", default="2012-2024")
    parser.add_argument("--idx", type=int, default=100)

    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)

    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--step", type=int, default=5)

    parser.add_argument("--output-name", default="stconvs2s_mapa_previsao.png")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    years = parse_years(args.years)

    dataset = RadarStationMemmapDataset(
        radar_root=args.dataset_path,
        years=years,
        t_in=args.t_in,
        t_out=args.t_out,
        stride=args.stride,
        split="test",
    )

    x, y, m = dataset[args.idx]

    input_size = (1, 3, args.t_in, args.height, args.width)

    model = STConvS2S_C(
        input_size=input_size,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        device=device,
        dropout_rate=args.dropout,
        step=args.step,
        output_channels=1,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    m = m.cpu().numpy()
    pred = pred.squeeze(0).cpu().numpy()

    radar_img = np.transpose(x[:, -1, :, :], (1, 2, 0))

    fig, axes = plt.subplots(args.t_out, 3, figsize=(11, 16))

    for t in range(args.t_out):
        obs = y[0, t]
        mask = m[0, t]
        obs_masked = np.where(mask == 1, obs, np.nan)
        pred_img = pred[0, t]

        axes[t, 0].imshow(radar_img)
        axes[t, 0].set_title("Radar entrada")
        axes[t, 0].axis("off")

        axes[t, 1].imshow(obs_masked, vmin=0, vmax=12.5)
        axes[t, 1].set_title(f"Observado T+{t+1}")
        axes[t, 1].axis("off")

        im2 = axes[t, 2].imshow(pred_img, vmin=0, vmax=12.5)
        axes[t, 2].set_title(f"STConvS2S T+{t+1}")
        axes[t, 2].axis("off")

    fig.colorbar(
        im2,
        ax=axes[:, 1:].ravel().tolist(),
        shrink=0.6,
        label="mm/15min",
    )

    output_path = out_dir / args.output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Figura salva em:", output_path)


if __name__ == "__main__":
    main()