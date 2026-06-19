from radar_station_memmap_dataset import RadarStationMemmapDataset

radar_root = "/home/noemi/atmoseer/data/datasets/radar_sumare_2022_2024_15min_256_por_ano"

dataset = RadarStationMemmapDataset(
    radar_root=radar_root,
    years=[2022],
    t_in=5,
    t_out=5,
    stride=5,
    split="train",
)

print("Total de amostras:", len(dataset))

x, y, m = dataset[0]

print("X:", x.shape)
print("Y:", y.shape)
print("M:", m.shape)

print("X min/max:", x.min().item(), x.max().item())
print("Y min/max:", y.min().item(), y.max().item())
print("M min/max:", m.min().item(), m.max().item())
print("M soma:", m.sum().item())

observed = y[m == 1]
print("Observações:", observed.numel())

if observed.numel() > 0:
    print("Y observado min/max:", observed.min().item(), observed.max().item())
else:
    print("Nenhuma observação nessa amostra.")