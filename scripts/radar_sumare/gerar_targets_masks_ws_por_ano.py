from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera Y_all e M_all das estações WebSirene alinhados aos timestamps do radar, por ano."
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
    print(f"GERANDO TARGETS/MÁSCARAS PARA {year}", flush=True)
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

        # Controle de qualidade:
        # remove valores fisicamente suspeitos
        df = df[df["m15"] <= 50.0]

        removed_qc += before_qc - len(df)

        # Alinha a chuva para a mesma grade temporal do radar: 15 min
        df["observation_datetime"] = (
            df["observation_datetime"]
            .dt.floor("15min")
        )

        all_rain.append(df)

    if not all_rain:
        print(f"[{year}] Nenhum dado WebSirene encontrado.", flush=True)
        return

    rain_df = pd.concat(all_rain, ignore_index=True)

    # Se após o floor houver leituras repetidas no mesmo horário,
    # mantém o maior valor de m15.
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

    # Converte pixel da grade original para 256x256
    rain_df["pixel_i_256"] = (
        rain_df["pixel_i"] * args.height / args.height_orig
    ).round().astype(int)

    rain_df["pixel_j_256"] = (
        rain_df["pixel_j"] * args.width / args.width_orig
    ).round().astype(int)

    rain_df["pixel_i_256"] = rain_df["pixel_i_256"].clip(0, args.height - 1)
    rain_df["pixel_j_256"] = rain_df["pixel_j_256"].clip(0, args.width - 1)

    Y_all = np.zeros(
        (len(radar_timestamps), args.height, args.width, 1),
        dtype=np.float32,
    )

    M_all = np.zeros(
        (len(radar_timestamps), args.height, args.width, 1),
        dtype=np.uint8,
    )

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

        # Se mais de uma estação cair no mesmo pixel/tempo, fica o maior valor.
        Y_all[t_idx, i, j, 0] = max(Y_all[t_idx, i, j, 0], rain)
        M_all[t_idx, i, j, 0] = 1

        used_rows += 1

    output_path = year_dir / "stations_targets_masks.npz"

    np.savez_compressed(
        output_path,
        Y_all=Y_all,
        M_all=M_all,
        timestamps=np.array(
            [ts.isoformat() for ts in radar_timestamps],
            dtype=object,
        ),
        metadata=np.array(
            {
                "year": year,
                "target": "m15",
                "target_unit": "mm/15min",
                "height": args.height,
                "width": args.width,
                "temporal_resolution_minutes": 15,
                "mask": "1 where station observation exists",
                "quality_control": "removed values above 50 mm/15min",
                "max_m15_allowed": 50.0,
            },
            dtype=object,
        ),
    )

    print(f"[{year}] Arquivo salvo em: {output_path}", flush=True)
    print(f"[{year}] Y_all: {Y_all.shape}", flush=True)
    print(f"[{year}] M_all: {M_all.shape}", flush=True)
    print(f"[{year}] Observações: {M_all.sum()}", flush=True)
    print(f"[{year}] Linhas usadas: {used_rows}", flush=True)
    print(f"[{year}] Linhas ignoradas por timestamp ausente: {skipped_rows}", flush=True)
    print(f"[{year}] Chuva máxima m15: {Y_all.max()}", flush=True)
    print(
        f"[{year}] Linhas removidas pelo controle de qualidade (>50 mm/15min): "
        f"{removed_qc}",
        flush=True,
    )


def main():
    args = parse_args()

    print("=== CONFIGURAÇÃO ===", flush=True)
    print("ws_root:", args.ws_root, flush=True)
    print("mapping:", args.mapping, flush=True)
    print("radar_root:", args.radar_root, flush=True)
    print("year_start:", args.year_start, flush=True)
    print("year_end:", args.year_end, flush=True)
    print("resize:", (args.height, args.width), flush=True)

    for year in range(args.year_start, args.year_end + 1):
        process_year(args, year)

    print("\nTODOS OS ANOS FINALIZADOS.", flush=True)


if __name__ == "__main__":
    main()