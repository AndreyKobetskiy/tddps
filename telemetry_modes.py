import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import median

from detect import CurrentHistory
from equipment import EQUIPMENT, lathe_current


MODES = {
    "lathe": ("cutting_mode", ("idle", "finishing", "roughing"), "load_pct"),
    "pump": ("pump_mode", ("off", "low_flow", "high_flow"), "flow_l_min"),
    "compressor": (
        "compressor_mode", ("off", "unloaded", "loaded"), "airflow_m3_min"
    ),
    "furnace": (
        "furnace_mode", ("cooling", "holding", "heating"), "heater_power_kw"
    ),
    "conveyor": (
        "conveyor_mode", ("stopped", "accumulating", "running"), "belt_speed_m_s"
    ),
    "chiller": (
        "chiller_mode", ("standby", "partial_load", "full_load"), "coolant_flow_l_min"
    ),
}

LEVELS = (0.10, 0.45, 0.80)
PART_STEPS = 10_000
EPOCH_MS = 1_767_225_600_000
TABLES = ("ground_truth", "observations", "anomaly_labels")


def csv_writer(stack, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
    writer = csv.writer(stream)
    writer.writerow(header)
    return writer


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def observer_masks(spec, rng):
    extra = list(range(3, len(spec.fields)))
    count = 3 if len(extra) == 5 else 4

    while True:
        masks = [tuple(sorted(rng.sample(extra, count)))
                 for _ in range(spec.observers)]
        if len(set(masks)) == spec.observers and set().union(*masks) == set(extra):
            return [(0, 1, 2, *mask) for mask in masks]


def reference_state(spec, machine_id, step):
    _, modes, _ = MODES[spec.kind]
    block, elapsed = divmod(step, 300)
    mode_index = (block + machine_id - 1) % len(modes)
    level = LEVELS[mode_index]
    previous = LEVELS[mode_index - 1] if block else level

    t = step / 10
    variation = 0.03 * math.sin(t / 75 + 0.7 * machine_id)
    slow_level = previous + (level - previous) * min(elapsed / 120, 1)

    values = []
    for i, (field, base, gain) in enumerate(zip(spec.fields, spec.base, spec.response)):
        if field == "current_a":
            values.append(lathe_current(values[spec.fields.index("load_pct")]))
        else:
            driver = slow_level if "_temp_" in field else level
            ripple = 0 if field == "load_pct" else 0.03 * math.sin(t / (3 + i))
            values.append(base + gain * (driver + variation + ripple))

    return values, modes[mode_index]


def generate(root, steps, seed):
    root.mkdir(parents=True)
    layout_rng = random.Random(seed)
    measurement_rng = random.Random(seed + 1)
    layouts = []
    next_observer_id = 1

    with ExitStack() as stack:
        machines = csv_writer(stack, root / "machines.csv", ("machine_id", "kind"))
        observers = csv_writer(
            stack, root / "observers.csv",
            ("observer_id", "machine_id", "observer_type", "observed_fields"),
        )
        for machine_id, spec in enumerate(EQUIPMENT, 1):
            machines.writerow((machine_id, spec.kind))
            mode_field = MODES[spec.kind][0]
            mode_readers = set(layout_rng.sample(range(spec.observers), 2))
            assignments = []
            for local_id, mask in enumerate(observer_masks(spec, layout_rng)):
                fields = [spec.fields[i] for i in mask]
                if local_id in mode_readers:
                    fields.append(mode_field)
                observers.writerow((
                    next_observer_id, machine_id, f"{spec.kind}_{local_id + 1}",
                    "|".join(fields),
                ))
                assignments.append((next_observer_id, mask, local_id in mode_readers))
                next_observer_id += 1
            layouts.append((machine_id, spec, assignments))

    anomaly_count = 0
    for part_no, start in enumerate(range(0, steps, PART_STEPS)):
        filename = f"part-{part_no:05d}.csv"
        for machine_id, spec, assignments in layouts:
            mode_field, modes, _ = MODES[spec.kind]
            columns = (*spec.fields, mode_field)
            with ExitStack() as stack:
                truth, observations, labels = [
                    csv_writer(stack, root / table / spec.kind / filename, header)
                    for table, header in zip(TABLES, (
                        ("machine_id", "timestamp_ms", *columns),
                        ("observer_id", "timestamp_ms", *columns),
                        ("observer_id", "timestamp_ms", "field", "delta", "before", "after"),
                    ))
                ]
                for step in range(start, min(start + PART_STEPS, steps)):
                    timestamp = EPOCH_MS + 100 * step
                    values, mode = reference_state(spec, machine_id, step)
                    truth.writerow((
                        machine_id, timestamp, *(f"{v:.4f}" for v in values), mode
                    ))

                    mode_index = len(spec.fields)
                    for observer, mask, has_mode in assignments:
                        readings = [""] * len(columns)
                        for i in mask:
                            value = values[i] + measurement_rng.gauss(0, spec.noise[i])
                            readings[i] = f"{value:.4f}"
                        if has_mode:
                            readings[mode_index] = mode

                        if measurement_rng.random() < 0.07:
                            available = (*mask, mode_index) if has_mode else mask
                            i = measurement_rng.choice(available)
                            if i == mode_index:
                                after = measurement_rng.choice([m for m in modes if m != mode])
                                readings[i] = after
                                labels.writerow((observer, timestamp, mode_field, "", mode, after))
                            else:
                                before = float(readings[i])
                                shift = (
                                    measurement_rng.choice((-1, 1))
                                    * measurement_rng.uniform(8, 16)
                                    * spec.noise[i]
                                )
                                after = round(before + shift, 4)
                                readings[i] = f"{after:.4f}"
                                labels.writerow((
                                    observer, timestamp, columns[i], f"{after - before:.4f}", "", ""
                                ))
                            anomaly_count += 1
                        observations.writerow((observer, timestamp, *readings))

    total = steps * (next_observer_id - 1)
    print(f"Наблюдений: {total:,}. Аномалий: {anomaly_count:,} ({anomaly_count / total:.2%})")


def event_key(row):
    return int(row["observer_id"]), int(row["timestamp_ms"]), row["field"]


def analyze(spec, batches, truth, labels):
    """Возвращает найденные тройки (наблюдатель, время, поле)."""
    mode_field, modes, signal = MODES[spec.kind]
    signal_index = spec.fields.index(signal)
    histories = defaultdict(CurrentHistory)
    predictions = set()

    for timestamp, rows in sorted(batches.items()):
        common = spec.fields[:3]
        centers = {field: median(float(row[field]) for row in rows) for field in common}
        demand = (centers[signal] - spec.base[signal_index]) / spec.response[signal_index]
        mode_index = min(range(len(LEVELS)), key=lambda i: abs(demand - LEVELS[i]))
        expected_mode = modes[mode_index]

        for row in rows:
            observer = int(row["observer_id"])
            for i, field in enumerate(common):
                score = abs(float(row[field]) - centers[field]) / (
                    math.sqrt(2) * spec.noise[i]
                )
                if score > 5:
                    predictions.add((observer, timestamp, field))

            if spec.kind == "lathe" and row["current_a"]:
                expected = lathe_current(centers["load_pct"])
                residual = float(row["current_a"]) - expected
                if histories[observer].score(timestamp, residual) > 5:
                    predictions.add((observer, timestamp, "current_a"))

            reported_mode = row[mode_field]
            if reported_mode and reported_mode != expected_mode:
                predictions.add((observer, timestamp, mode_field))

    return predictions


def visualize(root, output, parts, stride):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summary = []

    for spec in EQUIPMENT:
        mode_field, modes, _ = MODES[spec.kind]
        fields = (*spec.fields, mode_field)
        paths = sorted(
            (root / "observations" / spec.kind).glob("part-*.csv"),
            key=lambda p: int(p.stem.split("-")[1]),
        )[:parts]
        if not paths:
            raise ValueError(f"Нет CSV для {spec.kind} в {root}")

        truth = {}
        batches = defaultdict(list)
        labels = set()
        for path in paths:
            truth.update(
                (int(row["timestamp_ms"]), row)
                for row in read_rows(root / "ground_truth" / spec.kind / path.name)
            )
            for row in read_rows(path):
                batches[int(row["timestamp_ms"])].append(row)
            labels.update(
                event_key(row)
                for row in read_rows(root / "anomaly_labels" / spec.kind / path.name)
            )

        found = analyze(spec, batches, truth, labels)
        summary.append((spec.kind, fields, labels, found))
        origin = min(truth)
        timeline = sorted(truth)
        regular_times = set(timeline[::stride])

        fig, axes = plt.subplots(
            (len(fields) + 1) // 2, 2,
            figsize=(14, 3.4 * ((len(fields) + 1) // 2)),
            squeeze=False, layout="constrained",
        )
        fig.suptitle(spec.kind)

        for ax, field in zip(axes.flat, fields):
            categorical = field == mode_field
            if categorical:
                encode = {mode: i for i, mode in enumerate(modes)}.__getitem__
                ax.set_yticks(range(len(modes)), modes)
            else:
                encode = float

            times = timeline if categorical else timeline[::stride]
            xs = [(t - origin) / 1000 for t in times]
            ys = [encode(truth[t][field]) for t in times]
            if categorical:
                ax.step(xs, ys, where="post", color="#223344", label="Эталон")
            else:
                ax.plot(xs, ys, color="#223344", label="Эталон")

            samples = defaultdict(list)
            for t in timeline:
                for row in batches[t]:
                    value = row[field]
                    if not value:
                        continue
                    observer = int(row["observer_id"])
                    key = (observer, t, field)
                    if t in regular_times or key in labels or key in found:
                        samples[observer].append(((t - origin) / 1000, encode(value), key))

            for observer, points in sorted(samples.items()):
                ax.scatter(
                    [p[0] for p in points], [p[1] for p in points],
                    s=7, alpha=0.5, label=f"Наблюдатель {observer}",
                )
            marked = [p for points in samples.values() for p in points]
            marked_labels = [p for p in marked if p[2] in labels]
            marked_found = [p for p in marked if p[2] in found]
            if marked_labels:
                ax.scatter(
                    [p[0] for p in marked_labels], [p[1] for p in marked_labels],
                    marker="x", color="red", s=26, label="Внесённые искажения",
                )
            if marked_found:
                ax.scatter(
                    [p[0] for p in marked_found], [p[1] for p in marked_found],
                    facecolors="none", edgecolors="purple", s=46, label="Детектор",
                )
            ax.set_title(field)
            ax.set_xlabel("Время от начала, с")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)

        for ax in list(axes.flat)[len(fields):]:
            ax.set_visible(False)
        fig.savefig(output / f"{spec.kind}.png", dpi=115)
        plt.close(fig)

    fig, axes = plt.subplots(
        len(summary), figsize=(12, 3 * len(summary)), layout="constrained"
    )
    for ax, (kind, fields, labels, found) in zip(axes, summary):
        actual = Counter(f for _, _, f in labels)
        detected = Counter(f for _, _, f in labels & found)
        false_alarms = Counter(f for _, _, f in found - labels)
        xs = range(len(fields))
        ax.bar([x - 0.25 for x in xs], [actual[f] for f in fields], 0.25, label="В разметке")
        ax.bar(xs, [detected[f] for f in fields], 0.25, label="Найдены")
        ax.bar([x + 0.25 for x in xs], [false_alarms[f] for f in fields], 0.25,
               label="Срабатывания без метки")
        ax.set_xticks(list(xs), fields, rotation=24, ha="right")
        ax.set_title(kind)
        ax.set_ylabel("Количество")
        ax.legend()
    fig.savefig(output / "anomalies.png", dpi=115)
    plt.close(fig)
    print(f"Графики в {output.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    generation = sub.add_parser("generate")
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--steps", type=int, default=1000)
    generation.add_argument("--seed", type=int, default=42)

    plotting = sub.add_parser("plot")
    plotting.add_argument("dataset", type=Path)
    plotting.add_argument("--output", type=Path, default=Path("plots"))
    plotting.add_argument("--parts", type=int, default=1)
    plotting.add_argument("--stride", type=int, default=3)

    args = parser.parse_args()
    if args.command == "generate":
        if args.steps < 1 or args.output.exists():
            parser.error("Нужно положительное число шагов и новая папка для датасета.")
        generate(args.output, args.steps, args.seed)
    else:
        if args.parts < 1 or args.stride < 1:
            parser.error("--parts и --stride должны быть положительными.")
        visualize(args.dataset, args.output, args.parts, args.stride)


if __name__ == "__main__":
    main()