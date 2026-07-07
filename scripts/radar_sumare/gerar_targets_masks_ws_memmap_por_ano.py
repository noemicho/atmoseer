from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


MAX_M15 = 50.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera Y_all e M_all das estações WebSirene em memmap, alinhados aos timestamps do radar."
    )

    parser.add_argument("--ws-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)

    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)

    parser.add_argument("--station-start", type=int, default=1)
    parser.add_argument("--station-end", type=int, default=83)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)

    parser.add_argument("--height-orig", type=int, default=656)
    parser.add_argument("--width-orig", type=int, default=654)

    return parser.parse_args()


def normalize_ts(ts):
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def process_year(args, year):
    print("\n" + "=" * 60, flush=True)
    print(f"GERANDO Y/M MEMMAP PARA {year}", flush=True)
    print("=" * 60, flush=True)

    year_dir = args.radar_root / f"year={year}"
    radar_ts_path = year_dir / "radar_timestamps.npy"

    if not radar_ts_path.exists():
        print(f"[{year}] radar_timestamps.npy não encontrado. Pulando ano.", flush=True)
        return

    radar_timestamps = np.load(radar_ts_path, allow_pickle=True)
    radar_timestamps = [normalize_ts(ts) for ts in radar_timestamps]

    timestamp_to_idx = {ts: idx for idx, ts in enumerate(radar_timestamps)}

    print(f"[{year}] Timestamps do radar: {len(radar_timestamps)}", flush=True)

    mapping_df = pd.read_csv(args.mapping)

    all_rain = []
    removed_qc = 0

    for station_id in range(args.station_start, args.station_end + 1):
        path = (
            args.ws_root
            / f"station_id={station_id}"
            / f"year={year}"
            / "data.parquet"
        )

        if not path.exists():
            continue

        df = pd.read_parquet(path)

        df = df[
            [
                "id",
                "nome",
                "latitude",
                "longitude",
                "observation_datetime",
                "m15",
            ]
        ].copy()

        df = df.rename(columns={"id": "station_id"})

        df["observation_datetime"] = pd.to_datetime(
            df["observation_datetime"],
            utc=True,
        )

        df = df.dropna(subset=["m15"])

        before_qc = len(df)
        df = df[(df["m15"] >= 0) & (df["m15"] <= MAX_M15)]
        #df = df[df["m15"] <= MAX_M15]
        removed_qc += before_qc - len(df)

        df["observation_datetime"] = (
            df["observation_datetime"]
            .dt.floor("15min")
        )

        if len(df) > 0:
            all_rain.append(df)

    if not all_rain:
        print(f"[{year}] Nenhum dado WebSirene encontrado.", flush=True)
        return

    rain_df = pd.concat(all_rain, ignore_index=True)

    rain_df = rain_df.groupby(
        [
            "station_id",
            "nome",
            "latitude",
            "longitude",
            "observation_datetime",
        ],
        as_index=False,
    )["m15"].max()

    rain_df = rain_df.merge(
        mapping_df[["station_id", "pixel_i", "pixel_j"]],
        on="station_id",
        how="inner",
    )

    rain_df["pixel_i_256"] = (
        rain_df["pixel_i"] * args.height / args.height_orig
    ).round().astype(int)

    rain_df["pixel_j_256"] = (
        rain_df["pixel_j"] * args.width / args.width_orig
    ).round().astype(int)

    rain_df["pixel_i_256"] = rain_df["pixel_i_256"].clip(0, args.height - 1)
    rain_df["pixel_j_256"] = rain_df["pixel_j_256"].clip(0, args.width - 1)

    y_path = year_dir / "Y_all.dat"
    m_path = year_dir / "M_all.dat"
    metadata_path = year_dir / "targets_metadata.json"

    shape_y = (len(radar_timestamps), args.height, args.width, 1)

    Y_all = np.memmap(
        y_path,
        dtype=np.float32,
        mode="w+",
        shape=shape_y,
    )

    M_all = np.memmap(
        m_path,
        dtype=np.uint8,
        mode="w+",
        shape=shape_y,
    )

    Y_all[:] = 0.0
    M_all[:] = 0

    used_rows = 0
    skipped_rows = 0

    for _, row in rain_df.iterrows():
        ts = normalize_ts(row["observation_datetime"])

        if ts not in timestamp_to_idx:
            skipped_rows += 1
            continue

        t_idx = timestamp_to_idx[ts]

        i = int(row["pixel_i_256"])
        j = int(row["pixel_j_256"])

        rain = float(row["m15"])

        # transformação logarítmica
        rain = np.log1p(rain)

        Y_all[t_idx, i, j, 0] = max(Y_all[t_idx, i, j, 0], rain)
        M_all[t_idx, i, j, 0] = 1

        used_rows += 1

    Y_all.flush()
    M_all.flush()

    metadata = {
        "year": year,
        "target": "m15",
        "target_unit": "log1p(mm/15min)",
        "target_transform": "log1p",
        "original_target_unit": "mm/15min",
        "height": args.height,
        "width": args.width,
        "channels": 1,
        "temporal_resolution_minutes": 15,
        "shape": list(shape_y),
        "Y_file": "Y_all.dat",
        "M_file": "M_all.dat",
        "Y_dtype": "float32",
        "M_dtype": "uint8",
        "mask": "1 where station observation exists",
        "quality_control": "removed values above 50 mm/15min",
        "max_m15_allowed": MAX_M15,
        "removed_qc_rows": int(removed_qc),
        "used_rows": int(used_rows),
        "skipped_rows_timestamp_absent": int(skipped_rows),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[{year}] Arquivos salvos:", flush=True)
    print(f"  {y_path}", flush=True)
    print(f"  {m_path}", flush=True)
    print(f"  {metadata_path}", flush=True)
    print(f"[{year}] Shape Y/M: {shape_y}", flush=True)
    print(f"[{year}] Observações: {int(M_all.sum())}", flush=True)
    print(f"[{year}] Linhas usadas: {used_rows}", flush=True)
    print(f"[{year}] Linhas ignoradas por timestamp ausente: {skipped_rows}", flush=True)
    print(f"[{year}] Linhas removidas QC (>50 mm/15min): {removed_qc}", flush=True)
    print(f"[{year}] Máximo salvo em log1p: {float(Y_all.max()):.4f}", flush=True)
    print(f"[{year}] Equivalente em mm/15min: {float(np.expm1(Y_all.max())):.2f}", flush=True)

    del Y_all
    del M_all


def main():
    args = parse_args()

    print("=== CONFIGURAÇÃO ===", flush=True)
    print("ws_root:", args.ws_root, flush=True)
    print("mapping:", args.mapping, flush=True)
    print("radar_root:", args.radar_root, flush=True)
    print("year_start:", args.year_start, flush=True)
    print("year_end:", args.year_end, flush=True)
    print("resize:", (args.height, args.width), flush=True)
    print("MAX_M15:", MAX_M15, flush=True)

    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year)

    print("\nTODOS OS ANOS FINALIZADOS.", flush=True)


if __name__ == "__main__":
    main()