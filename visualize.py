"""Визуализация сгенерированного датасета телеметрии.

Примеры:
    python visualize.py sample --output plots
    python visualize.py data --output plots --partitions 2 --stride 5 \
        --field current_a --predictions predictions.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = (
    "speed_rpm", "load_pct", "motor_temp_c", "current_a", "voltage_v",
    "power_kw", "case_temp_c", "vibration_x_mm_s", "vibration_y_mm_s",
    "vibration_z_mm_s",
)
SIGMA = dict(zip(FIELDS, (3, 0.25, 0.12, 0.04, 0.25, 0.025, 0.10, 0.012, 0.012, 0.012)))
UNITS = {
    "speed_rpm": "об/мин", "load_pct": "%", "motor_temp_c": "°C",
    "current_a": "А", "voltage_v": "В", "power_kw": "кВт",
    "case_temp_c": "°C", "vibration_x_mm_s": "мм/с",
    "vibration_y_mm_s": "мм/с", "vibration_z_mm_s": "мм/с",
}
DEFAULT_FIELDS = ("speed_rpm", "load_pct", "motor_temp_c", "current_a")
THRESHOLD = 5.0
THIN = 4000          # точек на серию в скаттере


# --------------------------------------------------------------- чтение CSV
def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def load_catalogs(dataset):
    machines = {int(r["machine_id"]): r["name"]
                for r in read_rows(dataset / "machines.csv")}
    observers = {}
    for r in read_rows(dataset / "observers.csv"):
        oid = int(r["observer_id"])
        observers[oid] = (int(r["machine_id"]), r["observed_fields"].split("|"))
    return machines, observers


def partition_paths(dataset):
    return sorted((dataset / "observations").glob("part-*.csv"),
                  key=lambda p: int(p.stem.split("-")[1]))


def thin(xs, ys, limit=THIN):
    if len(xs) <= limit:
        return xs, ys
    step = len(xs) // limit + 1
    return xs[::step], ys[::step]


# ------------------------------------------------------------- сбор данных
def collect(dataset, observers, wanted_machines, parts, stride):
    truth = defaultdict(lambda: defaultdict(list))              # m -> f -> [(ts, v)]
    obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))    # m -> f -> oid -> [(ts, v)]
    labels = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # m -> f -> oid -> [(ts, delta)]

    for part in parts:
        kept_ts = set()
        # эталон
        step_idx = -1
        for ts_text, rows in groupby(read_rows(dataset / "ground_truth" / part.name),
                                     key=lambda r: r["timestamp_ms"]):
            step_idx += 1
            if step_idx % stride:
                continue
            ts = int(ts_text)
            for row in rows:
                m = int(row["machine_id"])
                if m not in wanted_machines:
                    continue
                kept_ts.add(ts)
                for f in FIELDS:
                    truth[m][f].append((ts, float(row[f])))
        # наблюдения
        step_idx = -1
        for ts_text, rows in groupby(read_rows(dataset / "observations" / part.name),
                                     key=lambda r: r["timestamp_ms"]):
            step_idx += 1
            if step_idx % stride:
                continue
            ts = int(ts_text)
            if ts not in kept_ts:
                continue
            for row in rows:
                oid = int(row["observer_id"])
                m = observers[oid][0]
                if m not in wanted_machines:
                    continue
                for f in FIELDS:
                    v = row[f]
                    if not v:
                        continue
                    obs[m][f][oid].append((ts, float(v)))
        # разметка
        label_path = dataset / "anomaly_labels" / part.name
        if label_path.is_file():
            for row in read_rows(label_path):
                ts = int(row["timestamp_ms"])
                if ts not in kept_ts:
                    continue
                oid = int(row["observer_id"])
                m = observers[oid][0]
                if m not in wanted_machines:
                    continue
                f = row["field"]
                labels[m][f][oid].append((ts, float(row["delta"])))
    return truth, obs, labels


def count_labels(dataset, parts):
    counts = Counter()
    for part in parts:
        lp = dataset / "anomaly_labels" / part.name
        if not lp.is_file():
            continue
        for r in read_rows(lp):
            counts[(int(r["observer_id"]), r["field"])] += 1
    return counts


def load_predictions(path):
    return [(int(r["observer_id"]), int(r["timestamp_ms"]), r["field"],
             float(r["score"]), r["method"]) for r in read_rows(path)]


def load_labels(dataset, parts):
    out = set()
    for part in parts:
        lp = dataset / "anomaly_labels" / part.name
        if not lp.is_file():
            continue
        for r in read_rows(lp):
            out.add((int(r["observer_id"]), int(r["timestamp_ms"]), r["field"]))
    return out


# --------------------------------------------------------------- графики
def plot_field(machine, machine_name, field, truth_m, obs_m, labels_m, path):
    pts_truth = truth_m[field]
    if not pts_truth:
        return
    t = [p[0] for p in pts_truth]
    v = [p[1] for p in pts_truth]
    truth_map = dict(pts_truth)
    sigma = SIGMA[field]

    fig, (ax_raw, ax_res) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax_raw.plot(t, v, color="black", lw=1.0, label="эталон")
    cmap = plt.cm.tab10

    for i, oid in enumerate(sorted(obs_m[field])):
        pts = obs_m[field][oid]
        if not pts:
            continue
        color = cmap(i % 10)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        sx, sy = thin(xs, ys)
        ax_raw.scatter(sx, sy, s=8, color=color, alpha=0.8, label=f"obs {oid}")

        rx = [p[0] for p in pts if p[0] in truth_map]
        ry = [p[1] - truth_map[p[0]] for p in pts if p[0] in truth_map]
        rx, ry = thin(rx, ry)
        ax_res.plot(rx, ry, color=color, lw=0.7, alpha=0.9)

        obs_map = dict(pts)
        anomaly_ts, anomaly_res = [], []
        for ts, _delta in labels_m.get(oid, []):
            if ts in truth_map and ts in obs_map:
                anomaly_ts.append(ts)
                anomaly_res.append(obs_map[ts] - truth_map[ts])
        if anomaly_ts:
            ax_res.scatter(anomaly_ts, anomaly_res, s=70, marker="x",
                           color="red", linewidths=2, zorder=6,
                           label="разметка" if i == 0 else None)

    ax_res.axhline(THRESHOLD * sigma, color="red", ls="--", lw=0.8, alpha=0.6)
    ax_res.axhline(-THRESHOLD * sigma, color="red", ls="--", lw=0.8, alpha=0.6)
    ax_raw.set_ylabel(f"{field}, {UNITS.get(field, '')}")
    ax_res.set_ylabel("остаток (набл. − эталон)")
    ax_res.set_xlabel("timestamp_ms")
    ax_raw.set_title(f"{machine_name}: {field}")
    ax_raw.legend(loc="upper right", fontsize=8, ncol=3)
    ax_res.legend(loc="upper right", fontsize=8)
    ax_raw.grid(alpha=0.3)
    ax_res.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_machine_overview(machine, machine_name, truth_m, path):
    fields = [f for f in FIELDS if truth_m.get(f)]
    if not fields:
        return
    cols = 2
    rows = (len(fields) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows),
                             squeeze=False, sharex=True)
    for i, f in enumerate(fields):
        ax = axes[i // cols][i % cols]
        t = [p[0] for p in truth_m[f]]
        v = [p[1] for p in truth_m[f]]
        ax.plot(t, v, lw=0.9)
        ax.set_title(f, fontsize=9)
        ax.set_ylabel(UNITS.get(f, ""), fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(len(fields), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(f"{machine_name} — эталонные сигналы")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_anomaly_heatmap(counts, observers, path):
    oids = sorted(observers)
    fields = list(FIELDS)
    matrix = np.zeros((len(fields), len(oids)))
    for (oid, f), c in counts.items():
        if oid in oids and f in fields:
            matrix[fields.index(f), oids.index(oid)] = c
    if matrix.max() == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(len(oids)))
    ax.set_xticklabels([str(o) for o in oids], fontsize=8)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=8)
    ax.set_xlabel("observer_id")
    ax.set_title("Внесённые искажения: поле × наблюдатель")
    vmax = matrix.max()
    for i in range(len(fields)):
        for j in range(len(oids)):
            val = int(matrix[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                        color="white" if val > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, label="кол-во")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scores_hist(preds, path):
    if not preds:
        return
    by_field = defaultdict(list)
    for _, _, f, s, _ in preds:
        by_field[f].append(s)
    fields = sorted(by_field)
    cols = 2
    rows = (len(fields) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), squeeze=False)
    for i, f in enumerate(fields):
        ax = axes[i // cols][i % cols]
        ax.hist(by_field[f], bins=50, log=True, color="steelblue")
        ax.axvline(THRESHOLD, color="red", ls="--", lw=0.8)
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("score")
        ax.grid(alpha=0.3)
    for j in range(len(fields), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle("Оценки детектора (лог-шкала по Y)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_precision_recall(preds, labels, path):
    stats = defaultdict(lambda: [0, 0, 0])  # field -> [tp, fp, fn]
    pred_set = set()
    for oid, ts, f, _s, _m in preds:
        key = (oid, ts, f)
        pred_set.add(key)
        stats[f][0 if key in labels else 1] += 1
    for key in labels:
        if key not in pred_set:
            stats[key[2]][2] += 1
    if not stats:
        return
    fields = sorted(stats)
    tp = [stats[f][0] for f in fields]
    fp = [stats[f][1] for f in fields]
    fn = [stats[f][2] for f in fields]
    x = np.arange(len(fields))
    w = 0.28
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, tp, w, label="TP", color="seagreen")
    ax.bar(x, fp, w, label="FP", color="indianred")
    ax.bar(x + w, fn, w, label="FN (не поймано)", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_title("Детектор vs. разметка по полям")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_index(output, plots):
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Telemetry plots</title>",
        "<style>body{font-family:sans-serif;max-width:1400px;margin:20px auto}"
        "img{max-width:100%;border:1px solid #ccc;margin-bottom:20px}</style>",
        "<h1>Графики телеметрии</h1>",
    ]
    for p in plots:
        lines.append(f"<h2>{p.name}</h2><img src='{p.name}'>")
    (output / "index.html").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", type=int, default=1,
                        help="сколько партиций читать (по умолчанию 1)")
    parser.add_argument("--stride", type=int, default=1,
                        help="брать каждый N-й шаг симуляции")
    parser.add_argument("--machine", type=int, action="append",
                        help="ограничить machine_id (можно повторять)")
    parser.add_argument("--field", action="append",
                        help="поле для подробных графиков (можно повторять)")
    parser.add_argument("--predictions", type=Path,
                        help="predictions.csv от detect.py")
    parser.add_argument("--no-overview", action="store_true",
                        help="не строить обзорные графики по станкам")
    args = parser.parse_args()

    if args.partitions <= 0 or args.stride <= 0:
        parser.error("--partitions и --stride должны быть положительными")
    if not (args.dataset / "observations").is_dir():
        parser.error("в каталоге нет observations/")

    machines, observers = load_catalogs(args.dataset)
    parts = partition_paths(args.dataset)
    if not parts:
        parser.error("не найдены партиции")
    parts = parts[:args.partitions]

    wanted_machines = set(args.machine) if args.machine else set(machines)
    bad = wanted_machines - set(machines)
    if bad:
        parser.error(f"неизвестные machine_id: {sorted(bad)}")

    fields = tuple(args.field) if args.field else DEFAULT_FIELDS
    bad = set(fields) - set(FIELDS)
    if bad:
        parser.error(f"неизвестные поля: {sorted(bad)}")

    args.output.mkdir(parents=True, exist_ok=True)
    plots = []

    print(f"читаю {len(parts)} партиций, stride={args.stride}")
    truth, obs, labels = collect(args.dataset, observers,
                                 wanted_machines, parts, args.stride)

    # 1. подробные графики
    for m in sorted(wanted_machines):
        for f in fields:
            if not truth[m][f]:
                continue
            path = args.output / f"machine{m:02d}_{f}.png"
            plot_field(m, machines[m], f, truth[m], obs[m], labels[m], path)
            plots.append(path)

    # 2. обзорные графики
    if not args.no_overview:
        for m in sorted(wanted_machines):
            path = args.output / f"machine{m:02d}_overview.png"
            plot_machine_overview(m, machines[m], truth[m], path)
            plots.append(path)

    # 3. теплокарта разметки
    counts = count_labels(args.dataset, parts)
    if counts:
        path = args.output / "anomalies_heatmap.png"
        plot_anomaly_heatmap(counts, observers, path)
        plots.append(path)

    # 4. детектор
    if args.predictions:
        preds = load_predictions(args.predictions)
        path = args.output / "detector_scores.png"
        plot_scores_hist(preds, path)
        plots.append(path)
        label_set = load_labels(args.dataset, parts)
        if label_set:
            path = args.output / "detector_pr.png"
            plot_precision_recall(preds, label_set, path)
            plots.append(path)

    write_index(args.output, plots)
    print(f"готово: {len(plots)} файлов в {args.output}")
    for p in plots:
        print(" ", p.name)


if __name__ == "__main__":
    main()