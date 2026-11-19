"""Generate reproducible machine telemetry in partitioned CSV files."""
import argparse
import csv
import math
import random
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "speed_rpm", "load_pct", "motor_temp_c", "current_a", "voltage_v",
    "power_kw", "case_temp_c", "vibration_x_mm_s", "vibration_y_mm_s",
    "vibration_z_mm_s",
)
NOISE = (3, 0.25, 0.12, 0.04, 0.25, 0.025, 0.10, 0.012, 0.012, 0.012)
EPOCH_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
PART_STEPS = 10_000


def observer_fields(rng):
    extra = list(range(3, len(FIELDS)))
    while True:
        groups = [tuple(sorted(rng.sample(extra, 4))) for _ in range(3)]
        if len(set(groups)) == 3 and set().union(*groups) == set(extra):
            return [(0, 1, 2, *group) for group in groups]


def machine_state(machine, step):
    seconds = step / 10
    phase = machine * 0.7
    load = 55 + 22 * math.sin(seconds / 75 + phase) + 8 * math.sin(seconds / 9)
    speed = 950 + 22 * load + 80 * math.sin(seconds / 23 + phase)
    voltage = 400 + 3 * math.sin(seconds / 17 + phase)
    current = 2.5 + 0.12 * load
    power = math.sqrt(3) * voltage * current * 0.86 / 1000
    motor_temp = 50 + 9 * math.sin(seconds / 75 + phase - 0.8) + machine
    case_temp = 37 + 6 * math.sin(seconds / 75 + phase - 1.3) + machine
    base_vibration = 0.6 + 0.018 * load
    vibration = [
        base_vibration * gain + 0.12 * math.sin(seconds / period + phase)
        for gain, period in ((1.0, 1.3), (0.8, 1.7), (0.65, 2.1))
    ]
    return speed, load, motor_temp, current, voltage, power, case_temp, *vibration


def csv_writer(stack, path, header):
    file = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(header)
    return writer


def write_catalogs(root, masks):
    with ExitStack() as stack:
        machines = csv_writer(stack, root / "machines.csv", ("machine_id", "name"))
        observers = csv_writer(
            stack, root / "observers.csv", ("observer_id", "machine_id", "observed_fields")
        )
        for machine, groups in enumerate(masks, 1):
            machines.writerow((machine, f"machine-{machine:02d}"))
            for index, fields in enumerate(groups, 1):
                observer = (machine - 1) * 3 + index
                observers.writerow((observer, machine, "|".join(FIELDS[i] for i in fields)))


def write_partition(root, part, start, count, masks, rng):
    suffix = f"part-{part:05d}.csv"
    truth_path = root / "ground_truth" / suffix
    observations_path = root / "observations" / suffix
    labels_path = root / "anomaly_labels" / suffix
    anomalous = 0
    with ExitStack() as stack:
        truth = csv_writer(stack, truth_path, ("machine_id", "timestamp_ms", *FIELDS))
        observations = csv_writer(
            stack, observations_path, ("observer_id", "timestamp_ms", *FIELDS)
        )
        labels = csv_writer(
            stack, labels_path, ("observer_id", "timestamp_ms", "field", "delta")
        )
        for step in range(start, start + count):
            timestamp = EPOCH_MS + 100 * step
            for machine, groups in enumerate(masks, 1):
                state = machine_state(machine, step)
                truth.writerow((machine, timestamp, *(f"{value:.4f}" for value in state)))
                for index, fields in enumerate(groups, 1):
                    observer = (machine - 1) * 3 + index
                    readings = [""] * len(FIELDS)
                    for field in fields:
                        measured = round(state[field] + rng.gauss(0, NOISE[field]), 4)
                        readings[field] = f"{measured:.4f}"
                    if rng.random() < 0.07:
                        field = rng.choice(fields)
                        before = float(readings[field])
                        shift = rng.choice((-1, 1)) * rng.uniform(8, 16) * NOISE[field]
                        after = round(before + shift, 4)
                        readings[field] = f"{after:.4f}"
                        labels.writerow((observer, timestamp, FIELDS[field], f"{after - before:.4f}"))
                        anomalous += 1
                    observations.writerow((observer, timestamp, *readings))
    size = sum(path.stat().st_size for path in (truth_path, observations_path, labels_path))
    return size, anomalous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--steps", type=int)
    mode.add_argument("--target-bytes", type=int)
    args = parser.parse_args()
    limit = args.steps if args.steps is not None else args.target_bytes
    if limit <= 0:
        parser.error("the requested limit must be positive")
    if args.output.exists():
        parser.error("output directory already exists")
    args.output.mkdir(parents=True)
    for table in ("ground_truth", "observations", "anomaly_labels"):
        (args.output / table).mkdir()
    configuration_rng = random.Random(args.seed)
    masks = [observer_fields(configuration_rng) for _ in range(6)]
    rng = random.Random(args.seed + 1)
    write_catalogs(args.output, masks)
    size = sum(path.stat().st_size for path in args.output.glob("*.csv"))
    step = part = anomalous = 0
    while True:
        count = PART_STEPS if args.steps is None else min(PART_STEPS, args.steps - step)
        part_size, part_anomalies = write_partition(args.output, part, step, count, masks, rng)
        size += part_size
        anomalous += part_anomalies
        step += count
        part += 1
        print(f"partitions={part} bytes={size:,} observations={step * 18:,} "
              f"anomalies={anomalous:,} ({anomalous / (step * 18):.2%})", flush=True)
        if (args.steps is not None and step == args.steps
                or args.target_bytes is not None and size >= args.target_bytes):
            break


if __name__ == "__main__":
    main()
