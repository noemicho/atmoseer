from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from radar_station_memmap_dataset import RadarStationMemmapDataset
from model.stconvs2s import STConvS2S_C


def parse_years(years_arg):
    if "-" in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y.strip()) for y in years_arg.split(",")]


def get_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(f"Coluna não encontrada. Opções: {options}. Colunas: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--years", default="2012-2024")
    parser.add_argument("--stations", default="10,11,12,13,14")
    parser.add_argument("--idx", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)

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

    parser.add_argument("--vmax", type=float, default=50.0)
    parser.add_argument("--mmh-equiv", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)
    station_ids = [int(s.strip()) for s in args.stations.split(",")]
    h = args.horizon - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    pred_np = pred.squeeze(0).cpu().numpy()

    radar_img = np.transpose(x_np[:, -1, :, :], (1, 2, 0))

    pred_map = np.clip(pred_np[0, h], 0, None)
    obs_map = y_np[0, h]

    print("=== DEBUG MAPA ===")
    print("obs min/max/mean:", obs_map.min(), obs_map.max(), obs_map.mean())
    print("pred bruto min/max/mean:", pred_np[0, h].min(), pred_np[0, h].max(), pred_np[0, h].mean())
    print("pred clip min/max/mean:", pred_map.min(), pred_map.max(), pred_map.mean())
    print("==================")

    factor = 4.0 if args.mmh_equiv else 1.0
    unit = "mm/h equivalente" if args.mmh_equiv else "mm/15min"

    pred_map = pred_map * factor
    obs_map = obs_map * factor

    mapping = pd.read_csv(args.mapping_csv)

    id_col = get_col(mapping, ["station_id", "id_estacao", "estacao", "id", "ID", "cod_estacao"])
    row_col = get_col(mapping, ["pixel_y", "row", "linha", "i", "y", "pixel_i"])
    col_col = get_col(mapping, ["pixel_x", "col", "coluna", "j", "x", "pixel_j"])

    for station_id in station_ids:
        station_row = mapping[mapping[id_col] == station_id]

        if station_row.empty:
            print(f"[AVISO] Estação {station_id} não encontrada.")
            continue

        py_orig = int(station_row.iloc[0][row_col])
        px_orig = int(station_row.iloc[0][col_col])

        orig_h = 654
        orig_w = 656

        # converte coordenadas da grade original para a grade 256x256 usada pela STConvS2S
        py = int(round(py_orig * (args.height - 1) / (orig_h - 1)))
        px = int(round(px_orig * (args.width - 1) / (orig_w - 1)))

        py = max(0, min(args.height - 1, py))
        px = max(0, min(args.width - 1, px))

        obs_value = obs_map[py, px]
        pred_value = pred_map[py, px]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        axes[0].imshow(radar_img, interpolation="nearest", origin="upper")
        axes[0].scatter(px, py, s=45, c="white", edgecolors="black", linewidths=0.8)
        axes[0].set_title("Radar RGB - última entrada")
        axes[0].axis("off")

        im = axes[1].imshow(
            pred_map,
            vmin=0,
            vmax=args.vmax,
            cmap="turbo",
            interpolation="bilinear",
            origin="upper",
        )
        axes[1].scatter(px, py, s=45, c="white", edgecolors="black", linewidths=0.8)
        axes[1].set_title(f"Previsão STConvS2S - T+{args.horizon}")
        axes[1].axis("off")

        axes[1].text(
            px + 5,
            py + 5,
            f"Obs: {obs_value:.2f} {unit}\nPrev: {pred_value:.2f} {unit}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )

        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(unit)

        fig.suptitle(f"Estação {station_id} - STConvS2S", fontsize=11)

        out_path = out_dir / f"stconvs2s_station_{station_id}_idx{args.idx}_tplus{args.horizon}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Figura salva:", out_path)


if __name__ == "__main__":
    main()