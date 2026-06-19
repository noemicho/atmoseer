from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera frames agregados do radar em memmap, separados por ano."
    )

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)

    parser.add_argument("--aggregate-minutes", type=int, default=15)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--min-frames-per-window", type=int, default=3)

    return parser.parse_args()


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def extract_timestamp(file_path: Path) -> datetime:
    stem = file_path.stem.rstrip("_")
    parts = stem.split("_")

    if len(parts) != 5:
        raise ValueError(f"Nome fora do padrão esperado: {file_path.name}")

    year, month, day, hour, minute = map(int, parts)
    return datetime(year, month, day, hour, minute)


def score_file(file_path: Path) -> int:
    return 0 if file_path.stem.endswith("_") else 1


def collect_files_for_year(data_root: Path, year: int):
    best_by_timestamp = {}

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    for current_day in daterange(start_date, end_date):
        if current_day.day == 1:
            print(
                f"[{year}] Lendo mês {current_day.month:02d}...",
                flush=True,
            )

        day_dir = (
            data_root
            / f"{current_day.year:04d}"
            / f"{current_day.month:02d}"
            / f"{current_day.day:02d}"
        )

        if not day_dir.exists():
            continue

        files = sorted(
            f for f in day_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".png"
        )

        for f in files:
            try:
                ts = extract_timestamp(f)
            except Exception:
                continue

            if ts not in best_by_timestamp:
                best_by_timestamp[ts] = f
            elif score_file(f) > score_file(best_by_timestamp[ts]):
                best_by_timestamp[ts] = f

    return sorted(best_by_timestamp.items(), key=lambda x: x[0])


def load_radar_image_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGBA")
        arr = np.array(img, dtype=np.uint8)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    mask = alpha > 0
    rgb = rgb * mask[:, :, None]

    rgb_img = Image.fromarray(rgb, mode="RGB")
    rgb_img = rgb_img.resize((width, height), Image.NEAREST)

    return np.array(rgb_img, dtype=np.uint8)


def build_time_buckets(file_ts_list, aggregate_minutes: int):
    buckets = defaultdict(list)

    for ts, path in file_ts_list:
        minute_bucket = (ts.minute // aggregate_minutes) * aggregate_minutes
        bucket_ts = ts.replace(minute=minute_bucket, second=0, microsecond=0)
        buckets[bucket_ts].append(path)

    return buckets


def process_year(
    year: int,
    data_root: Path,
    output_root: Path,
    aggregate_minutes: int,
    height: int,
    width: int,
    min_frames_per_window: int,
):
    print("\n" + "=" * 60, flush=True)
    print(f"INICIANDO ANO {year}", flush=True)
    print("=" * 60, flush=True)

    output_dir = output_root / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_path = output_dir / "radar_frames.dat"
    timestamps_path = output_dir / "radar_timestamps.npy"
    metadata_path = output_dir / "metadata.json"

    file_ts_list = collect_files_for_year(data_root, year)

    print(f"[{year}] Total de PNGs válidos: {len(file_ts_list)}", flush=True)

    if len(file_ts_list) == 0:
        print(f"[{year}] Nenhum PNG encontrado. Pulando ano.", flush=True)
        return

    buckets = build_time_buckets(
        file_ts_list=file_ts_list,
        aggregate_minutes=aggregate_minutes,
    )

    valid_bucket_items = [
        (bucket_ts, paths)
        for bucket_ts, paths in sorted(buckets.items(), key=lambda x: x[0])
        if len(paths) >= min_frames_per_window
    ]

    n_frames_estimado = len(valid_bucket_items)

    print(
        f"[{year}] Total estimado de frames agregados de 15 min: {n_frames_estimado}",
        flush=True,
    )

    frames_mm = np.memmap(
        frames_path,
        dtype=np.uint8,
        mode="w+",
        shape=(n_frames_estimado, height, width, 3),
    )

    timestamps = []
    write_idx = 0
    invalid_count = 0
    ignored_windows = 0

    for idx, (bucket_ts, paths) in enumerate(valid_bucket_items):
        agg_frame = None
        valid_frames = 0

        for path in paths:
            try:
                frame = load_radar_image_rgb(
                    path,
                    width=width,
                    height=height,
                )
            except Exception as e:
                invalid_count += 1

                if invalid_count <= 20:
                    print(
                        f"[{year}] [AVISO] Ignorando imagem inválida: {path} | erro: {e}",
                        flush=True,
                    )
                continue

            if agg_frame is None:
                agg_frame = frame
            else:
                agg_frame = np.maximum(agg_frame, frame)

            valid_frames += 1

        if valid_frames < min_frames_per_window:
            ignored_windows += 1

            if ignored_windows <= 20:
                print(
                    f"[{year}] [AVISO] Janela ignorada {bucket_ts}: "
                    f"apenas {valid_frames} frames válidos",
                    flush=True,
                )
            continue

        frames_mm[write_idx] = agg_frame
        timestamps.append(bucket_ts.isoformat())
        write_idx += 1

        if idx % 1000 == 0:
            print(
                f"[{year}] Processados {idx}/{n_frames_estimado} janelas | "
                f"gravados {write_idx} frames | "
                f"imagens inválidas {invalid_count} | "
                f"janelas ignoradas {ignored_windows}",
                flush=True,
            )

    frames_mm.flush()

    real_size = write_idx * height * width * 3
    
    with open(frames_path, "r+b") as f:
        f.truncate(real_size)

    n_frames_final = write_idx

    np.save(
        timestamps_path,
        np.array(timestamps, dtype=object),
    )

    metadata = {
        "year": year,
        "aggregate_minutes": aggregate_minutes,
        "aggregation_method": "max",
        "height": height,
        "width": width,
        "channels": 3,
        "dtype": "uint8",
        "shape": [n_frames_final, height, width, 3],
        "frames_file": "radar_frames.dat",
        "timestamps_file": "radar_timestamps.npy",
        "invalid_images": invalid_count,
        "ignored_windows": ignored_windows,
        "estimated_frames": n_frames_estimado,
        "final_frames": n_frames_final,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[{year}] FINALIZADO", flush=True)
    print(f"[{year}] frames finais: {n_frames_final}", flush=True)
    print(f"[{year}] imagens inválidas: {invalid_count}", flush=True)
    print(f"[{year}] janelas ignoradas: {ignored_windows}", flush=True)
    print(f"[{year}] salvo em: {output_dir}", flush=True)


def main():
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("=== CONFIGURAÇÃO GERAL ===", flush=True)
    print("data_root:", args.data_root, flush=True)
    print("output_root:", args.output_root, flush=True)
    print("year_start:", args.year_start, flush=True)
    print("year_end:", args.year_end, flush=True)
    print("aggregate_minutes:", args.aggregate_minutes, flush=True)
    print("resize:", (args.height, args.width), flush=True)

    for year in range(args.year_start, args.year_end + 1):
        process_year(
            year=year,
            data_root=args.data_root,
            output_root=args.output_root,
            aggregate_minutes=args.aggregate_minutes,
            height=args.height,
            width=args.width,
            min_frames_per_window=args.min_frames_per_window,
        )

    print("\nTODOS OS ANOS FINALIZADOS.", flush=True)


if __name__ == "__main__":
    main()