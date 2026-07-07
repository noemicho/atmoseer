from pathlib import Path
import argparse
import numpy as np

from radar_station_memmap_dataset import RadarStationMemmapDataset


def parse_years(years_arg):
    if "-" in years_arg:
        start, end = years_arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y.strip()) for y in years_arg.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-root", type=str, required=True)
    parser.add_argument("--years", type=str, default="2012-2024")
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--t-out", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    years = parse_years(args.years)

    dataset = RadarStationMemmapDataset(
        radar_root=args.radar_root,
        years=years,
        t_in=args.t_in,
        t_out=args.t_out,
        stride=args.stride,
        split=args.split,
    )

    se_total = 0.0
    ae_total = 0.0
    bias_total = 0.0
    n_total = 0

    se_h = np.zeros(args.t_out)
    ae_h = np.zeros(args.t_out)
    bias_h = np.zeros(args.t_out)
    n_h = np.zeros(args.t_out)

    bins = [
        ("Fraca (<5 mm/h equiv.)", 0.0, 1.25),
        ("Moderada (5-25 mm/h equiv.)", 1.25, 6.25),
        ("Forte (25-50 mm/h equiv.)", 6.25, 12.5),
        ("Extrema (>50 mm/h equiv.)", 12.5, np.inf),
    ]

    class_stats = {
        label: {"se": 0.0, "ae": 0.0, "bias": 0.0, "n": 0}
        for label, _, _ in bins
    }

    for idx, (year, start_idx) in enumerate(dataset.samples):
        data = dataset.year_data[year]
        Y_all = data["Y_all"]
        M_all = data["M_all"]

        current_idx = start_idx + args.t_in - 1
        y_start = start_idx + args.t_in
        y_end = y_start + args.t_out

        # chuva observada no último instante de entrada
        y_current = np.expm1(Y_all[current_idx:current_idx + 1])
        m_current = M_all[current_idx:current_idx + 1]

        # chuva futura real
        y_true = np.expm1(Y_all[y_start:y_end])
        m_future = M_all[y_start:y_end]

        # persistência: repete o valor atual nos 5 passos futuros
        y_pred = np.repeat(y_current, args.t_out, axis=0)

        # só avalia onde existe observação atual e futura
        m_eval = m_future * np.repeat(m_current, args.t_out, axis=0)

        valid = m_eval == 1
        if valid.sum() == 0:
            continue

        err = y_pred - y_true

        se_total += np.sum((err[valid]) ** 2)
        ae_total += np.sum(np.abs(err[valid]))
        bias_total += np.sum(err[valid])
        n_total += valid.sum()

        for h in range(args.t_out):
            valid_h = valid[h]
            if valid_h.sum() == 0:
                continue

            err_h = err[h][valid_h]

            se_h[h] += np.sum(err_h ** 2)
            ae_h[h] += np.sum(np.abs(err_h))
            bias_h[h] += np.sum(err_h)
            n_h[h] += valid_h.sum()

        for label, low, high in bins:
            if np.isinf(high):
                class_mask = valid & (y_true >= low)
            else:
                class_mask = valid & (y_true >= low) & (y_true < high)

            n = class_mask.sum()
            if n == 0:
                continue

            err_c = err[class_mask]

            class_stats[label]["se"] += np.sum(err_c ** 2)
            class_stats[label]["ae"] += np.sum(np.abs(err_c))
            class_stats[label]["bias"] += np.sum(err_c)
            class_stats[label]["n"] += n

        if (idx + 1) % 1000 == 0:
            print(f"Processadas {idx + 1}/{len(dataset.samples)} amostras", flush=True)

    print("\n===== PERSISTÊNCIA: MÉTRICAS GLOBAIS =====")
    print(f"N: {n_total}")
    print(f"RMSE: {np.sqrt(se_total / n_total):.4f}")
    print(f"MAE: {ae_total / n_total:.4f}")
    print(f"Bias: {bias_total / n_total:.4f}")

    print("\n===== PERSISTÊNCIA: POR HORIZONTE =====")
    for h in range(args.t_out):
        print(
            f"T+{h+1}: "
            f"N={int(n_h[h])}, "
            f"RMSE={np.sqrt(se_h[h] / n_h[h]):.4f}, "
            f"MAE={ae_h[h] / n_h[h]:.4f}, "
            f"Bias={bias_h[h] / n_h[h]:.4f}"
        )

    print("\n===== PERSISTÊNCIA: POR INTENSIDADE =====")
    for label, stats in class_stats.items():
        n = stats["n"]
        print(
            f"{label}: "
            f"n={n}, "
            f"RMSE={np.sqrt(stats['se'] / n):.4f}, "
            f"MAE={stats['ae'] / n:.4f}, "
            f"Bias={stats['bias'] / n:.4f}"
        )


if __name__ == "__main__":
    main()