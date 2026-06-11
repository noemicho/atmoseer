from __future__ import annotations

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


# ==========================================================
# 1) PARÂMETROS DE LINHA DE COMANDO
# ==========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dataset seq2seq RGB a partir de PNGs do radar com agregação temporal."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Raiz dos PNGs do radar. Ex: /home/noemi/atmoseer/data/radar_sumare",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Arquivo .npz de saída",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Data inicial no formato YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Data final no formato YYYY-MM-DD",
    )
    ####
    parser.add_argument(
        "--raw-time-step-minutes",
        type=int,
        default=2,
        help="Resolução temporal original dos arquivos, em minutos",
    )
    parser.add_argument(
        "--aggregate-minutes",
        type=int,
        default=15,
        help="Resolução temporal final após agregação, em minutos, por máximo",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Altura final das imagens",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Largura final das imagens",
    )
    parser.add_argument(
        "--t-in",
        type=int,
        default=5,
        help="Quantidade de frames de entrada",
    )
    parser.add_argument(
        "--t-out",
        type=int,
        default=5,
        help="Quantidade de frames de saída",
    )
    parser.add_argument(
        "--small-sample",
        action="store_true",
        help="Usa uma pequena amostra para teste rápido",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limite opcional de arquivos válidos a processar",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Passo entre amostras consecutivas. Ex: 1 usa todas; 5 pula de 5 em 5.",
    )

    return parser.parse_args()


# ==========================================================
# 2) FUNÇÕES AUXILIARES DE DATA
# ==========================================================
def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# ==========================================================
# 3) EXTRAÇÃO DE TIMESTAMP DO NOME DO ARQUIVO
# ==========================================================
def extract_timestamp(file_path: Path) -> datetime:
    """
    Espera nomes como:
    2024_03_06_12_34.png
    2024_03_06_12_34_.png
    """
    stem = file_path.stem.rstrip("_")
    parts = stem.split("_")

    if len(parts) != 5:
        raise ValueError(f"Nome fora do padrão esperado: {file_path.name}")

    year, month, day, hour, minute = map(int, parts)
    return datetime(year, month, day, hour, minute)


def score_file(file_path: Path) -> int:
    """
    Se houver duplicata do mesmo timestamp, preferimos o arquivo
    sem underscore no final.
    """
    return 0 if file_path.stem.endswith("_") else 1


# ==========================================================
# 4) COLETA DOS ARQUIVOS NO PERÍODO ESCOLHIDO
# ==========================================================
def collect_files_in_period(
    data_root: Path,
    start_date: date,
    end_date: date,
    max_files: int | None = None
) -> List[Tuple[datetime, Path]]:
    """
    Percorre o período escolhido e coleta PNGs válidos.
    Mantém apenas o melhor arquivo por timestamp.
    """
    best_by_timestamp = {}

    for current_day in daterange(start_date, end_date):
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
            else:
                if score_file(f) > score_file(best_by_timestamp[ts]):
                    best_by_timestamp[ts] = f

            if max_files is not None and len(best_by_timestamp) >= max_files:
                break

        if max_files is not None and len(best_by_timestamp) >= max_files:
            break

    cleaned = sorted(best_by_timestamp.items(), key=lambda x: x[0])
    return cleaned


# ==========================================================
# 5) SEPARAÇÃO EM BLOCOS TEMPORAIS CONTÍNUOS
# ==========================================================
def split_into_continuous_blocks(
    file_ts_list: List[Tuple[datetime, Path]],
    expected_delta: timedelta
) -> List[List[Tuple[datetime, Path]]]:
    """
    Separa a série em blocos contínuos.
    Exemplo: se a resolução esperada é 2 min, qualquer gap maior
    quebra o bloco.
    """
    if not file_ts_list:
        return []

    blocks = []
    current_block = [file_ts_list[0]]

    for i in range(1, len(file_ts_list)):
        prev_ts, _ = file_ts_list[i - 1]
        curr_ts, _ = file_ts_list[i]

        if (curr_ts - prev_ts) == expected_delta:
            current_block.append(file_ts_list[i])
        else:
            blocks.append(current_block)
            current_block = [file_ts_list[i]]

    blocks.append(current_block)
    return blocks


# ==========================================================
# 6) LEITURA E PRÉ-PROCESSAMENTO DA IMAGEM RGB
# ==========================================================
def load_radar_image_rgb(
    path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Lê PNG RGBA, remove o fundo transparente com alpha
    e devolve uma imagem RGB [H, W, 3] em uint8.
    """
    img = Image.open(path).convert("RGBA")
    arr = np.array(img).astype(np.uint8)   # [H, W, 4]

    rgb = arr[:, :, :3]   # [H, W, 3]
    alpha = arr[:, :, 3]  # [H, W]

    # máscara do que é dado real do radar
    mask = (alpha > 0).astype(np.uint8)

    # remove fundo transparente
    rgb = rgb * mask[:, :, None]

    # redimensiona
    rgb_img = Image.fromarray(rgb, mode="RGB")
    rgb_img = rgb_img.resize((width, height), Image.BILINEAR)

    rgb_resized = np.array(rgb_img, dtype=np.uint8)  # [H, W, 3]
    return rgb_resized

################### importante!
def aggregate_block_by_time_window(
    block: List[Tuple[datetime, Path]],
    aggregate_minutes: int,
    width: int,
    height: int,
) -> List[Tuple[datetime, np.ndarray]]:
    """
    Agrega PNGs em janelas temporais.

    Exemplo:
    aggregate_minutes = 15

    Frames de 2 em 2 minutos dentro da mesma janela de 15 min
    viram um único frame usando máximo pixel a pixel.
    """
    grouped = {}

    for ts, path in block:
        minute_bucket = (ts.minute // aggregate_minutes) * aggregate_minutes
        bucket_ts = ts.replace(minute=minute_bucket, second=0, microsecond=0)

        try:
            frame = load_radar_image_rgb(path, width=width, height=height)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[WARN] Ignorando imagem inválida: {path} | erro: {e}")
            continue

        if bucket_ts not in grouped:
            grouped[bucket_ts] = []

        grouped[bucket_ts].append(frame)

    aggregated = []

    for bucket_ts in sorted(grouped.keys()):
        ## filtrar janelas com poucos frames -> pouca informação
        if len(grouped[bucket_ts]) < 3:
            continue

        frames = np.stack(grouped[bucket_ts], axis=0)

        # Máximo pixel a pixel
        agg_frame = np.max(frames, axis=0).astype(np.uint8)

        aggregated.append((bucket_ts, agg_frame))

    return aggregated



# ==========================================================
# 7) GERAÇÃO DAS SEQUÊNCIAS T_IN -> T_OUT
# ==========================================================
def create_sequences_from_block(
    block: List[Tuple[datetime, Path]],
    t_in: int,
    t_out: int,
    width: int,
    height: int,
    aggregate_minutes: int,
    stride: int,
) -> Tuple[np.ndarray | None, np.ndarray | None, List[List[str]] | None]:

    aggregated = aggregate_block_by_time_window(
        block=block,
        aggregate_minutes=aggregate_minutes,
        width=width,
        height=height,
    )

    if not aggregated:
        return None, None, None

    timestamps = [ts for ts, _ in aggregated]
    frames = [frame for _, frame in aggregated]

    frames = np.stack(frames, axis=0).astype(np.uint8)

    n_possible_samples = len(frames) - (t_in + t_out) + 1

    if n_possible_samples <= 0:
        return None, None, None

    # Aqui entra o stride:
    # stride=1 -> cria uma amostra começando em cada frame
    # stride=5 -> cria uma amostra a cada 5 frames
    sample_indices = list(range(0, n_possible_samples, stride))
    n_samples = len(sample_indices)

    H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]

    X = np.empty((n_samples, t_in, H, W, C), dtype=np.uint8)
    Y = np.empty((n_samples, t_out, H, W, C), dtype=np.uint8)
    Y_ts_list: List[List[str]] = []

    for j, i in enumerate(sample_indices):
        X[j] = frames[i:i + t_in]
        Y[j] = frames[i + t_in:i + t_in + t_out]

        Y_ts_list.append([
            ts.isoformat()
            for ts in timestamps[i + t_in:i + t_in + t_out]
        ])

    return X, Y, Y_ts_list


# ==========================================================
# 8) FUNÇÃO PRINCIPAL
# ==========================================================
def main():
    args = parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    if end_date < start_date:
        raise ValueError("end-date deve ser maior ou igual a start-date")

    expected_delta = timedelta(minutes=args.raw_time_step_minutes)

    print("=== CONFIGURAÇÃO ===")
    print("data_root:", args.data_root)
    print("output:", args.output)
    print("start_date:", start_date)
    print("end_date:", end_date)
    print("raw_time_step_minutes:", args.raw_time_step_minutes)
    print("aggregate_minutes:", args.aggregate_minutes)
    print("resize:", (args.height, args.width))
    print("t_in:", args.t_in)
    print("t_out:", args.t_out)
    print("small_sample:", args.small_sample)
    print("max_files:", args.max_files)
    print("stride:", args.stride)

    max_files = args.max_files
    if args.small_sample and max_files is None:
        max_files = 300

    file_ts_list = collect_files_in_period(
        data_root=args.data_root,
        start_date=start_date,
        end_date=end_date,
        max_files=max_files,
    )

    print(f"\nTotal de timestamps únicos válidos no período: {len(file_ts_list)}")

    if not file_ts_list:
        print("Nenhum arquivo válido encontrado no período.")
        return

    #raw_blocks = split_into_continuous_blocks(file_ts_list, expected_delta)
    raw_blocks = [file_ts_list]

    print(f"Quantidade de blocos contínuos na resolução original: {len(raw_blocks)}")

    all_X = []
    all_Y = []
    all_Y_ts = []

    for i, block in enumerate(raw_blocks, start=1):
        print(f"\nBloco {i}: {len(block)} frames brutos")

        X_block, Y_block, Y_ts_block = create_sequences_from_block(
            block=block,
            t_in=args.t_in,
            t_out=args.t_out,
            width=args.width,
            height=args.height,
            aggregate_minutes=args.aggregate_minutes,
            stride=args.stride,
        )

        if X_block is None:
            print("  Bloco pequeno demais após agregação para gerar sequências.")
            continue

        print(f"  X_block shape: {X_block.shape} | dtype: {X_block.dtype}")
        print(f"  Y_block shape: {Y_block.shape} | dtype: {Y_block.dtype}")

        all_X.append(X_block)
        all_Y.append(Y_block)
        all_Y_ts.extend(Y_ts_block)

    if not all_X:
        print("Nenhuma sequência foi gerada.")
        return

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("\n===== DATASET FINAL =====")
    print("X shape:", X.shape, "| dtype:", X.dtype)
    print("Y shape:", Y.shape, "| dtype:", Y.dtype)
    print("Número de amostras:", len(all_Y_ts))

    np.savez_compressed(
        args.output,
        X=X,
        Y=Y,
        y_timestamps=np.array(all_Y_ts, dtype=object),
        metadata=np.array(
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "raw_time_step_minutes": args.raw_time_step_minutes,
                "aggregate_minutes": args.aggregate_minutes,
                "aggregation_method": "max",
                "height": args.height,
                "width": args.width,
                "t_in": args.t_in,
                "t_out": args.t_out,
                "channels": 3,
                "mode": "rgb",
                "stride": args.stride,
            },
            dtype=object
        ),
    )

    print(f"\nDataset salvo em:\n{args.output}")


if __name__ == "__main__":
    main()