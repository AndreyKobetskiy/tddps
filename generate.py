"""Generate reproducible machine telemetry in partitioned CSV files.

Six machines of six distinct physical types.  Types differ in:
  - numeric parameters (nominal speed / load / voltage / thermal / vibration),
  - the set of observable properties (`active_fields`), and
  - categorical attributes: static (`category`, `drive_type`, `cooling`,
    `criticality`) and dynamic (`status`), whose label set is type-specific.

The shared electromechanical model `current = 2.5 + 0.12 * load_pct` and
`power = sqrt(3) * voltage * current * 0.86 / 1000` is kept across all types
so `detect.py` stays valid without modification.  Every type observes
`current_a`; `detect.py` also handles empty `current_a` gracefully.
"""
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
COMMON_INDICES = (0, 1, 2)  # speed_rpm, load_pct, motor_temp_c — always observed
NOISE = (3, 0.25, 0.12, 0.04, 0.25, 0.025, 0.10, 0.012, 0.012, 0.012)
EPOCH_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
PART_STEPS = 10_000


MACHINE_TYPES = (
    {   # 1 — CNC mill: full rotating-sensor suite.
        "slug": "cnc-mill",
        "category": "milling",
        "drive_type": "spindle",
        "cooling": "liquid",
        "criticality": "high",
        "active_fields": ("current_a", "voltage_v", "power_kw",
                          "vibration_x_mm_s", "vibration_y_mm_s"),
        "status_thresholds": (20.0, 30.0),
        "status_labels": ("idle", "warmup", "running"),
        "speed_base": 1000.0, "speed_load": 22.0, "speed_ripple": 90.0, "speed_period": 23.0,
        "load_base": 55.0, "load_amp1": 22.0, "load_period1": 75.0,
        "load_amp2": 8.0,  "load_period2": 9.0,
        "voltage_base": 400.0, "voltage_ripple": 3.0, "voltage_period": 17.0,
        "motor_temp_base": 50.0, "motor_temp_amp": 9.0,  "motor_temp_lag": 0.8,
        "case_temp_base": 37.0,  "case_temp_amp": 6.0,   "case_temp_lag": 1.3,
        "vib_base": 0.60, "vib_load": 0.018,
        "vib_gains":  (1.00, 0.80, 0.65),
        "vib_periods": (1.3, 1.7, 2.1),
        "vib_phases": (0.0, 0.7, 1.3),
    },
    {   # 2 — Lathe: rotating + single-axis vibration.
        "slug": "lathe",
        "category": "turning",
        "drive_type": "spindle",
        "cooling": "liquid",
        "criticality": "medium",
        "active_fields": ("current_a", "voltage_v", "power_kw", "vibration_x_mm_s"),
        "status_thresholds": (25.0, 35.0),
        "status_labels": ("idle", "warmup", "running"),
        "speed_base": 1400.0, "speed_load": 14.0, "speed_ripple": 40.0, "speed_period": 19.0,
        "load_base": 48.0, "load_amp1": 30.0, "load_period1": 48.0,
        "load_amp2": 5.0,  "load_period2": 6.5,
        "voltage_base": 400.0, "voltage_ripple": 2.0, "voltage_period": 23.0,
        "motor_temp_base": 44.0, "motor_temp_amp": 11.0, "motor_temp_lag": 1.1,
        "case_temp_base": 33.0,  "case_temp_amp": 8.0,   "case_temp_lag": 1.6,
        "vib_base": 0.45, "vib_load": 0.030,
        "vib_gains":  (1.20, 0.55, 0.70),
        "vib_periods": (1.1, 2.4, 1.8),
        "vib_phases": (0.3, 1.1, 0.5),
    },
    {   # 3 — Hydraulic press: only Z-vibration is informative.
        "slug": "hydraulic-press",
        "category": "pressing",
        "drive_type": "hydraulic",
        "cooling": "passive",
        "criticality": "high",
        "active_fields": ("current_a", "voltage_v", "vibration_z_mm_s"),
        "status_thresholds": (40.0, 70.0),
        "status_labels": ("releasing", "holding", "pressing"),
        "speed_base": 300.0, "speed_load": 6.0, "speed_ripple": 10.0, "speed_period": 31.0,
        "load_base": 82.0, "load_amp1": 12.0, "load_period1": 120.0,
        "load_amp2": 3.0,  "load_period2": 14.0,
        "voltage_base": 405.0, "voltage_ripple": 4.0, "voltage_period": 37.0,
        "motor_temp_base": 58.0, "motor_temp_amp": 6.0,  "motor_temp_lag": 2.2,
        "case_temp_base": 41.0,  "case_temp_amp": 4.0,   "case_temp_lag": 2.8,
        "vib_base": 1.10, "vib_load": 0.008,
        "vib_gains":  (0.60, 0.90, 1.30),
        "vib_periods": (0.6, 0.9, 1.4),
        "vib_phases": (1.0, 0.2, 1.6),
    },
    {   # 4 — Screw compressor: rich electrical + thermal + X/Y vibration.
        "slug": "screw-compressor",
        "category": "compressing",
        "drive_type": "direct",
        "cooling": "liquid",
        "criticality": "high",
        "active_fields": ("current_a", "voltage_v", "power_kw", "case_temp_c",
                          "vibration_x_mm_s", "vibration_y_mm_s"),
        "status_thresholds": (30.0, 45.0),
        "status_labels": ("unloading", "loading", "loaded"),
        "speed_base": 3600.0, "speed_load": 3.0, "speed_ripple": 15.0, "speed_period": 41.0,
        "load_base": 70.0, "load_amp1": 5.0, "load_period1": 300.0,
        "load_amp2": 1.5,  "load_period2": 22.0,
        "voltage_base": 400.0, "voltage_ripple": 1.5, "voltage_period": 29.0,
        "motor_temp_base": 63.0, "motor_temp_amp": 4.0,  "motor_temp_lag": 3.1,
        "case_temp_base": 46.0,  "case_temp_amp": 3.0,   "case_temp_lag": 3.6,
        "vib_base": 0.80, "vib_load": 0.005,
        "vib_gains":  (1.10, 1.10, 1.10),
        "vib_periods": (0.4, 0.6, 0.8),
        "vib_phases": (0.0, 1.0, 2.0),
    },
    {   # 5 — Conveyor drive: minimal sensor suite.
        "slug": "conveyor-drive",
        "category": "conveying",
        "drive_type": "belt",
        "cooling": "air",
        "criticality": "low",
        "active_fields": ("current_a", "voltage_v"),
        "status_thresholds": (10.0, 25.0),
        "status_labels": ("stopped", "ramping", "running"),
        "speed_base": 250.0, "speed_load": 4.0, "speed_ripple": 5.0, "speed_period": 60.0,
        "load_base": 28.0, "load_amp1": 18.0, "load_period1": 90.0,
        "load_amp2": 6.0,  "load_period2": 7.0,
        "voltage_base": 400.0, "voltage_ripple": 2.5, "voltage_period": 33.0,
        "motor_temp_base": 38.0, "motor_temp_amp": 5.0,  "motor_temp_lag": 0.6,
        "case_temp_base": 29.0,  "case_temp_amp": 4.0,   "case_temp_lag": 1.0,
        "vib_base": 0.30, "vib_load": 0.010,
        "vib_gains":  (1.30, 0.50, 0.40),
        "vib_periods": (2.0, 3.5, 4.2),
        "vib_phases": (0.4, 1.5, 0.9),
    },
    {   # 6 — Centrifugal pump: Y/Z vibration dominates.
        "slug": "centrifugal-pump",
        "category": "pumping",
        "drive_type": "direct",
        "cooling": "liquid",
        "criticality": "medium",
        "active_fields": ("current_a", "voltage_v", "power_kw",
                          "vibration_y_mm_s", "vibration_z_mm_s"),
        "status_thresholds": (35.0, 55.0),
        "status_labels": ("idle", "starting", "pumping"),
        "speed_base": 1750.0, "speed_load": 6.0, "speed_ripple": 20.0, "speed_period": 27.0,
        "load_base": 62.0, "load_amp1": 20.0, "load_period1": 55.0,
        "load_amp2": 7.0,  "load_period2": 8.0,
        "voltage_base": 398.0, "voltage_ripple": 3.5, "voltage_period": 21.0,
        "motor_temp_base": 47.0, "motor_temp_amp": 10.0, "motor_temp_lag": 0.9,
        "case_temp_base": 35.0,  "case_temp_amp": 7.0,   "case_temp_lag": 1.5,
        "vib_base": 0.55, "vib_load": 0.022,
        "vib_gains":  (0.70, 1.40, 0.90),
        "vib_periods": (1.6, 1.2, 2.5),
        "vib_phases": (0.8, 0.1, 1.4),
    },
)


def make_observer_masks(rng, spec):
    """Three distinct observer field sets for one machine of the given type.

    Common fields (0, 1, 2) are always present.  Extras are drawn from the
    type's `active_fields`; the three observer subsets must be distinct and
    their union must equal the type's active extras.
    """
    extras = [FIELDS.index(name) for name in spec["active_fields"]]
    if len(extras) < 2:
        raise ValueError(
            f"type {spec['slug']!r} needs >=2 active extra fields, got {len(extras)}"
        )
    rng.shuffle(extras)
    if len(extras) == 2:
        groups = [[extras[0]], [extras[1]], list(extras)]
    else:
        groups = [[extras[0]], [extras[1]], [extras[2]]]
        for idx in extras[3:]:
            groups[rng.randint(0, 2)].append(idx)
    return [tuple(sorted((*COMMON_INDICES, *g))) for g in groups]


def machine_spec(machine):
    return MACHINE_TYPES[(machine - 1) % len(MACHINE_TYPES)]


def machine_status(spec, load_pct):
    idle_below, warmup_below = spec["status_thresholds"]
    idle_label, warmup_label, running_label = spec["status_labels"]
    if load_pct < idle_below:
        return idle_label
    if load_pct < warmup_below:
        return warmup_label
    return running_label


def machine_state(spec, step, phase):
    seconds = step / 10.0
    load = (
        spec["load_base"]
        + spec["load_amp1"] * math.sin(seconds / spec["load_period1"] + phase)
        + spec["load_amp2"] * math.sin(seconds / spec["load_period2"])
    )
    speed = (
        spec["speed_base"]
        + spec["speed_load"] * load
        + spec["speed_ripple"] * math.sin(seconds / spec["speed_period"] + phase)
    )
    voltage = spec["voltage_base"] + spec["voltage_ripple"] * math.sin(
        seconds / spec["voltage_period"] + phase
    )
    # Shared electromechanical model -> detect.py's load->current stays exact.
    current = 2.5 + 0.12 * load
    power = math.sqrt(3) * voltage * current * 0.86 / 1000.0
    motor_temp = spec["motor_temp_base"] + spec["motor_temp_amp"] * math.sin(
        seconds / spec["load_period1"] + phase - spec["motor_temp_lag"]
    )
    case_temp = spec["case_temp_base"] + spec["case_temp_amp"] * math.sin(
        seconds / spec["load_period1"] + phase - spec["case_temp_lag"]
    )
    base_vib = spec["vib_base"] + spec["vib_load"] * load
    vibration = [
        base_vib * gain + 0.12 * math.sin(seconds / period + phase + vib_phase)
        for gain, period, vib_phase in zip(
            spec["vib_gains"], spec["vib_periods"], spec["vib_phases"]
        )
    ]
    return speed, load, motor_temp, current, voltage, power, case_temp, *vibration


def csv_writer(stack, path, header):
    file = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(header)
    return writer


MACHINE_COLUMNS = (
    "machine_id", "name", "category", "drive_type", "cooling",
    "criticality", "active_fields",
)


def write_catalogs(root, specs, masks):
    with ExitStack() as stack:
        machines = csv_writer(stack, root / "machines.csv", MACHINE_COLUMNS)
        observers = csv_writer(
            stack, root / "observers.csv",
            ("observer_id", "machine_id", "observed_fields"),
        )
        for machine, (spec, groups) in enumerate(zip(specs, masks), 1):
            machines.writerow((
                machine,
                f"{spec['slug']}-{machine:02d}",
                spec["category"],
                spec["drive_type"],
                spec["cooling"],
                spec["criticality"],
                "|".join(spec["active_fields"]),
            ))
            for index, fields in enumerate(groups, 1):
                observer = (machine - 1) * 3 + index
                observers.writerow((
                    observer, machine, "|".join(FIELDS[i] for i in fields)
                ))


def write_partition(root, part, start, count, specs, masks, rng):
    suffix = f"part-{part:05d}.csv"
    truth_path = root / "ground_truth" / suffix
    observations_path = root / "observations" / suffix
    labels_path = root / "anomaly_labels" / suffix
    anomalous = 0
    with ExitStack() as stack:
        truth = csv_writer(
            stack, truth_path, ("machine_id", "timestamp_ms", *FIELDS, "status")
        )
        observations = csv_writer(
            stack, observations_path,
            ("observer_id", "timestamp_ms", *FIELDS, "status"),
        )
        labels = csv_writer(
            stack, labels_path, ("observer_id", "timestamp_ms", "field", "delta")
        )
        for step in range(start, start + count):
            timestamp = EPOCH_MS + 100 * step
            for machine, (spec, groups) in enumerate(zip(specs, masks), 1):
                phase = (machine - 1) * 0.7
                state = machine_state(spec, step, phase)
                status = machine_status(spec, state[1])
                truth.writerow((
                    machine, timestamp,
                    *(f"{value:.4f}" for value in state), status,
                ))
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
                        labels.writerow((
                            observer, timestamp, FIELDS[field], f"{after - before:.4f}"
                        ))
                        anomalous += 1
                    observations.writerow((observer, timestamp, *readings, status))
    size = sum(path.stat().st_size for path in (truth_path, observations_path, labels_path))
    return size, anomalous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--machines", type=int, default=len(MACHINE_TYPES),
                        help="number of machines; types are cycled")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--steps", type=int)
    mode.add_argument("--target-bytes", type=int)
    args = parser.parse_args()
    limit = args.steps if args.steps is not None else args.target_bytes
    if limit <= 0:
        parser.error("the requested limit must be positive")
    if args.machines <= 0:
        parser.error("--machines must be positive")
    if args.output.exists():
        parser.error("output directory already exists")
    args.output.mkdir(parents=True)
    for table in ("ground_truth", "observations", "anomaly_labels"):
        (args.output / table).mkdir()
    configuration_rng = random.Random(args.seed)
    specs = [machine_spec(m) for m in range(1, args.machines + 1)]
    masks = [make_observer_masks(configuration_rng, spec) for spec in specs]
    rng = random.Random(args.seed + 1)
    write_catalogs(args.output, specs, masks)
    size = sum(path.stat().st_size for path in args.output.glob("*.csv"))
    step = part = anomalous = 0
    n_observers = args.machines * 3
    while True:
        count = PART_STEPS if args.steps is None else min(PART_STEPS, args.steps - step)
        part_size, part_anomalies = write_partition(
            args.output, part, step, count, specs, masks, rng
        )
        size += part_size
        anomalous += part_anomalies
        step += count
        part += 1
        print(
            f"partitions={part} bytes={size:,} observations={step * n_observers:,} "
            f"anomalies={anomalous:,} ({anomalous / (step * n_observers):.2%})",
            flush=True,
        )
        if (args.steps is not None and step == args.steps
                or args.target_bytes is not None and size >= args.target_bytes):
            break


if __name__ == "__main__":
    main()