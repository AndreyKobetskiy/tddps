"""Detect anomalous readings in a generated tddps dataset."""
import argparse
import csv
from collections import defaultdict, deque
from itertools import groupby
from math import hypot, sqrt
from pathlib import Path
from statistics import median

from equipment import EQUIPMENT, lathe_current

LATHE = EQUIPMENT[0]
COMMON = LATHE.fields[:3]
SUPPORTED = (*COMMON, "current_a")
SIGMA = dict(zip(LATHE.fields, LATHE.noise))
CURRENT_SIGMA = hypot(SIGMA["current_a"], 0.12 * SIGMA["load_pct"])
THRESHOLD = 5


class CurrentHistory:
    def __init__(self):
        self.residuals = deque()
        self.center = 0.0
        self.scale = CURRENT_SIGMA
        self.next_update = 0

    def score(self, timestamp, residual):
        while self.residuals and self.residuals[0][0] < timestamp - 60_000:
            self.residuals.popleft()
        if timestamp >= self.next_update:
            if len(self.residuals) >= 20:
                values = [value for _, value in self.residuals]
                self.center = median(values)
                mad = median(abs(value - self.center) for value in values)
                self.scale = max(CURRENT_SIGMA, 1.4826 * mad)
            else:
                self.center = 0.0
                self.scale = CURRENT_SIGMA
            self.next_update = timestamp + 1000
        score = abs(residual - self.center) / self.scale
        self.residuals.append((timestamp, residual))
        return score


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def detect_partition(path, machines, histories, spec):
    common = spec.fields[:3]
    scales = {field: sqrt(2) * noise for field, noise in zip(common, spec.noise)}
    expected_observers = spec.observers
    for timestamp_text, rows in groupby(read_rows(path), key=lambda row: row["timestamp_ms"]):
        timestamp = int(timestamp_text)
        groups = defaultdict(list)
        for row in rows:
            observer = int(row["observer_id"])
            groups[machines[observer]].append((observer, row))
        for readings in groups.values():
            if (len({observer for observer, _ in readings}) != expected_observers
                    or len(readings) != expected_observers):
                raise ValueError(f"Expected {expected_observers} distinct observers at {timestamp}")
            centers = {
                field: median(float(row[field]) for _, row in readings)
                for field in common
            }
            expected_current = None
            if spec.kind == "lathe":
                expected_current = lathe_current(centers["load_pct"])
            for observer, row in readings:
                for field in common:
                    score = abs(float(row[field]) - centers[field]) / scales[field]
                    if score > THRESHOLD:
                        yield (observer, timestamp, field), score, "peer_median"
                if expected_current is not None and row["current_a"]:
                    residual = float(row["current_a"]) - expected_current
                    score = histories[observer].score(timestamp, residual)
                    if score > THRESHOLD:
                        yield (observer, timestamp, "current_a"), score, "load_residual"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    observations = args.dataset / "observations"
    parts = []
    for spec in EQUIPMENT:
        folder = observations / spec.kind
        if folder.is_dir():
            parts.extend((part, spec) for part in folder.glob("part-*.csv"))
    if not parts:
        parts = [(part, LATHE) for part in observations.glob("part-*.csv")]
    parts.sort(key=lambda entry: (int(entry[0].stem.split("-")[1]), entry[1].kind))
    if not parts:
        parser.error("no observation partitions found")
    label_dir = args.dataset / "anomaly_labels"
    if args.evaluate and any(
        not (label_dir / part.relative_to(observations)).is_file() for part, _ in parts
    ):
        parser.error("evaluation requires matching label partitions")
    machines = {
        int(row["observer_id"]): int(row["machine_id"])
        for row in read_rows(args.dataset / "observers.csv")
    }

    histories = defaultdict(CurrentHistory)
    predicted = true_positive = supported_labels = all_labels = 0
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("observer_id", "timestamp_ms", "field", "score", "method"))
        for part, spec in parts:
            labels = set()
            if args.evaluate:
                labels = {
                    (int(row["observer_id"]), int(row["timestamp_ms"]), row["field"])
                    for row in read_rows(label_dir / part.relative_to(observations))
                }
                all_labels += len(labels)
                fields = SUPPORTED if spec.kind == "lathe" else spec.fields[:3]
                supported_labels += sum(field in fields for _, _, field in labels)
            for key, score, method in detect_partition(part, machines, histories, spec):
                writer.writerow((*key, f"{score:.4f}", method))
                predicted += 1
                true_positive += key in labels

    print(f"Predictions: {predicted}")
    if args.evaluate:
        print(f"True positives: {true_positive}, supported labels: {supported_labels}, "
              f"all labels: {all_labels}")
        for name, total in (
            ("Precision", predicted),
            ("Recall (supported fields)", supported_labels),
            ("Recall (all fields)", all_labels),
        ):
            value = f"{true_positive / total:.2%}" if total else "n/a"
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
