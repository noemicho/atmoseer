from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

"""
python gerar_targets_masks_alertario_memmap_por_ano.py \
  --alertario-root /home/noemi/atmoseer/data/alertario/pluviometricos_parquet \
  --mapping /home/noemi/atmoseer/scripts/radar_sumare/mapeamento_pixel_estacao_alertario.csv \
  --radar-root /home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --year-start 2012 \
  --year-end 2024
"""


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Gera targets e máscaras das estações do Alerta Rio "
            "em memmap, alinhados aos timestamps do radar."
        )
    )

    parser.add_argument(
        "--alertario-root",
        type=Path,
        required=True,
        help="Pasta contendo os arquivos Parquet do Alerta Rio.",
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="CSV com o mapeamento das estações para os pixels.",
    )

    parser.add_argument(
        "--radar-root",
        type=Path,
        required=True,
        help="Pasta contendo os dados do radar organizados por ano.",
    )

    parser.add_argument(
        "--year-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--year-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--height-orig",
        type=int,
        default=656,
    )

    parser.add_argument(
        "--width-orig",
        type=int,
        default=654,
    )

    return parser.parse_args()


def normalize_ts(ts):

    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def load_mapping(args):

    mapping_df = pd.read_csv(args.mapping)

    # Permite que o CSV use tanto station_id quanto estacao_id.
    if (
        "station_id" not in mapping_df.columns
        and "estacao_id" in mapping_df.columns
    ):

        mapping_df = mapping_df.rename(
            columns={
                "estacao_id": "station_id",
            }
        )

    required_columns = [
        "station_id",
        "pixel_i",
        "pixel_j",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in mapping_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "O arquivo de mapeamento não possui as colunas: "
            f"{missing_columns}"
        )

    mapping_df = mapping_df[
        required_columns
    ].copy()

    mapping_df = mapping_df.dropna(
        subset=required_columns
    )

    mapping_df["station_id"] = mapping_df[
        "station_id"
    ].astype(int)

    if mapping_df["station_id"].duplicated().any():

        duplicated_ids = (
            mapping_df.loc[
                mapping_df["station_id"].duplicated(
                    keep=False
                ),
                "station_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existem identificadores duplicados no mapeamento: "
            f"{duplicated_ids}"
        )

    mapping_df["pixel_i_256"] = (
        mapping_df["pixel_i"]
        * args.height
        / args.height_orig
    ).round().astype(int)

    mapping_df["pixel_j_256"] = (
        mapping_df["pixel_j"]
        * args.width
        / args.width_orig
    ).round().astype(int)

    mapping_df["pixel_i_256"] = (
        mapping_df["pixel_i_256"]
        .clip(
            0,
            args.height - 1,
        )
    )

    mapping_df["pixel_j_256"] = (
        mapping_df["pixel_j_256"]
        .clip(
            0,
            args.width - 1,
        )
    )

    return mapping_df[
        [
            "station_id",
            "pixel_i_256",
            "pixel_j_256",
        ]
    ]


def process_year(args, year, parquet_files, mapping_df):

    print(
        "\n" + "=" * 60,
        flush=True,
    )

    print(
        f"GERANDO TARGETS E MÁSCARAS ALERTA RIO: {year}",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    year_dir = (
        args.radar_root
        / f"year={year}"
    )

    radar_ts_path = (
        year_dir
        / "radar_timestamps.npy"
    )

    if not radar_ts_path.exists():

        print(
            f"[{year}] radar_timestamps.npy não encontrado. "
            "Pulando ano.",
            flush=True,
        )

        return

    radar_timestamps = np.load(
        radar_ts_path,
        allow_pickle=True,
    )

    radar_timestamps = [
        normalize_ts(ts)
        for ts in radar_timestamps
    ]

    timestamp_to_idx = {
        ts: idx
        for idx, ts in enumerate(
            radar_timestamps
        )
    }

    print(
        f"[{year}] Timestamps do radar: "
        f"{len(radar_timestamps)}",
        flush=True,
    )

    shape_y = (
        len(radar_timestamps),
        args.height,
        args.width,
        1,
    )

    # Nomes separados para não sobrescrever os dados da WebSirene.
    y_path = (
        year_dir
        / "Y_alertario.dat"
    )

    m_path = (
        year_dir
        / "M_alertario.dat"
    )

    metadata_path = (
        year_dir
        / "targets_alertario_metadata.json"
    )

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

    year_start = pd.Timestamp(
        year=year,
        month=1,
        day=1,
        tz="UTC",
    )

    year_end = pd.Timestamp(
        year=year + 1,
        month=1,
        day=1,
        tz="UTC",
    )

    total_rows_year = 0

    removed_invalid_datetime = 0
    removed_invalid_m15 = 0

    removed_without_mapping = 0
    skipped_rows_timestamp_absent = 0

    used_rows = 0

    files_with_year_data = 0

    for file_idx, path in enumerate(
        parquet_files,
        start=1,
    ):

        try:

            df = pd.read_parquet(
                path,
                columns=[
                    "estacao_id",
                    "dia_utc",
                    "m15",
                ],
            )

        except Exception as error:

            print(
                f"[{year}] Erro ao ler {path.name}: "
                f"{error}",
                flush=True,
            )

            continue

        df = df.rename(
            columns={
                "estacao_id": "station_id",
                "dia_utc": "observation_datetime",
            }
        )

        df["observation_datetime"] = pd.to_datetime(
            df["observation_datetime"],
            utc=True,
            errors="coerce",
        )

        invalid_datetime = (
            df["observation_datetime"]
            .isna()
            .sum()
        )

        removed_invalid_datetime += int(
            invalid_datetime
        )

        df = df.dropna(
            subset=[
                "observation_datetime",
            ]
        )

        df = df[
            (
                df["observation_datetime"]
                >= year_start
            )
            & (
                df["observation_datetime"]
                < year_end
            )
        ].copy()

        if df.empty:
            continue

        files_with_year_data += 1

        total_rows_year += len(df)

        df["m15"] = pd.to_numeric(
            df["m15"],
            errors="coerce",
        )

        valid_m15 = (
            df["m15"].notna()
            & (df["m15"] >= 0)
        )

        removed_invalid_m15 += int(
            (~valid_m15).sum()
        )

        df = df.loc[
            valid_m15
        ].copy()

        if df.empty:
            continue

        df = df.dropna(
            subset=[
                "station_id",
            ]
        )

        if df.empty:
            continue

        df["station_id"] = df[
            "station_id"
        ].astype(int)

        df["observation_datetime"] = (
            df["observation_datetime"]
            .dt.floor("15min")
        )

        # Se houver mais de um registro da mesma estação
        # no mesmo horário, mantém o maior valor.
        df = (
            df.groupby(
                [
                    "station_id",
                    "observation_datetime",
                ],
                as_index=False,
            )["m15"]
            .max()
        )

        rows_before_mapping = len(df)

        df = df.merge(
            mapping_df,
            on="station_id",
            how="inner",
            validate="many_to_one",
        )

        removed_without_mapping += (
            rows_before_mapping
            - len(df)
        )

        if df.empty:
            continue

        df["t_idx"] = (
            df["observation_datetime"]
            .map(timestamp_to_idx)
        )

        valid_timestamp = (
            df["t_idx"]
            .notna()
        )

        skipped_rows_timestamp_absent += int(
            (~valid_timestamp).sum()
        )

        df = df.loc[
            valid_timestamp
        ].copy()

        if df.empty:
            continue

        t_idx = (
            df["t_idx"]
            .to_numpy(
                dtype=np.int64
            )
        )

        pixel_i = (
            df["pixel_i_256"]
            .to_numpy(
                dtype=np.int64
            )
        )

        pixel_j = (
            df["pixel_j_256"]
            .to_numpy(
                dtype=np.int64
            )
        )

        rain = (
            df["m15"]
            .to_numpy(
                dtype=np.float32
            )
        )

        rain_log = np.log1p(
            rain
        ).astype(
            np.float32
        )

        # Preserva o maior valor quando diferentes estações,
        # ou diferentes arquivos, atingem o mesmo pixel/horário.
        np.maximum.at(
            Y_all,
            (
                t_idx,
                pixel_i,
                pixel_j,
                np.zeros(
                    len(df),
                    dtype=np.int64,
                ),
            ),
            rain_log,
        )

        M_all[
            t_idx,
            pixel_i,
            pixel_j,
            0,
        ] = 1

        used_rows += len(df)

        print(
            f"[{year}] "
            f"Arquivo {file_idx}/{len(parquet_files)} | "
            f"{path.name} | "
            f"linhas utilizadas: {len(df)}",
            flush=True,
        )

    Y_all.flush()
    M_all.flush()

    metadata = {
        "source": "alertario",
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

        "Y_file": y_path.name,
        "M_file": m_path.name,

        "Y_dtype": "float32",
        "M_dtype": "uint8",

        "mask": (
            "1 where an Alerta Rio station observation exists"
        ),

        "quality_control": (
            "removed missing and negative precipitation values"
        ),

        "max_m15_allowed": None,

        "total_input_files": len(
            parquet_files
        ),

        "files_with_year_data": int(
            files_with_year_data
        ),

        "total_rows_year": int(
            total_rows_year
        ),

        "invalid_datetime_rows_scanned": int(
            removed_invalid_datetime
        ),

        "removed_invalid_m15_rows": int(
            removed_invalid_m15
        ),

        "removed_without_mapping_rows": int(
            removed_without_mapping
        ),

        "used_rows": int(
            used_rows
        ),

        "skipped_rows_timestamp_absent": int(
            skipped_rows_timestamp_absent
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\n[{year}] Arquivos salvos:",
        flush=True,
    )

    print(
        f"  {y_path}",
        flush=True,
    )

    print(
        f"  {m_path}",
        flush=True,
    )

    print(
        f"  {metadata_path}",
        flush=True,
    )

    print(
        f"[{year}] Shape Y/M: {shape_y}",
        flush=True,
    )

    print(
        f"[{year}] Arquivos com dados do ano: "
        f"{files_with_year_data}",
        flush=True,
    )

    print(
        f"[{year}] Linhas do ano: "
        f"{total_rows_year}",
        flush=True,
    )

    print(
        f"[{year}] Linhas usadas: "
        f"{used_rows}",
        flush=True,
    )

    print(
        f"[{year}] Valores inválidos de m15 removidos: "
        f"{removed_invalid_m15}",
        flush=True,
    )

    print(
        f"[{year}] Registros sem estação mapeada: "
        f"{removed_without_mapping}",
        flush=True,
    )

    print(
        f"[{year}] Registros sem timestamp correspondente: "
        f"{skipped_rows_timestamp_absent}",
        flush=True,
    )

    print(
        f"[{year}] Observações na máscara: "
        f"{int(M_all.sum())}",
        flush=True,
    )

    max_log = float(
        Y_all.max()
    )

    print(
        f"[{year}] Máximo salvo em log1p: "
        f"{max_log:.4f}",
        flush=True,
    )

    print(
        f"[{year}] Equivalente em mm/15min: "
        f"{float(np.expm1(max_log)):.2f}",
        flush=True,
    )

    if used_rows == 0:

        print(
            f"[{year}] ATENÇÃO: nenhum registro do Alerta Rio "
            "foi alinhado aos timestamps do radar.",
            flush=True,
        )

    del Y_all
    del M_all


def main():

    args = parse_args()

    if args.year_start > args.year_end:
        raise ValueError(
            "--year-start não pode ser maior que --year-end."
        )

    parquet_files = sorted(
        args.alertario_root.rglob(
            "*.parquet"
        )
    )

    if not parquet_files:
        raise ValueError(
            "Nenhum arquivo Parquet encontrado em: "
            f"{args.alertario_root}"
        )

    mapping_df = load_mapping(
        args
    )

    print(
        "=== CONFIGURAÇÃO ===",
        flush=True,
    )

    print(
        "alertario_root:",
        args.alertario_root,
        flush=True,
    )

    print(
        "mapping:",
        args.mapping,
        flush=True,
    )

    print(
        "radar_root:",
        args.radar_root,
        flush=True,
    )

    print(
        "year_start:",
        args.year_start,
        flush=True,
    )

    print(
        "year_end:",
        args.year_end,
        flush=True,
    )

    print(
        "resize:",
        (
            args.height,
            args.width,
        ),
        flush=True,
    )

    print(
        "arquivos_parquet:",
        len(parquet_files),
        flush=True,
    )

    print(
        "estacoes_mapeadas:",
        len(mapping_df),
        flush=True,
    )

    for year in range(
        args.year_start,
        args.year_end + 1,
    ):

        process_year(
            args,
            year,
            parquet_files,
            mapping_df,
        )

    print(
        "\nTODOS OS ANOS FINALIZADOS.",
        flush=True,
    )


if __name__ == "__main__":
    main()