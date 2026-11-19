"""Detect anomalous readings in a generated tddps dataset."""
import argparse
import csv
from collections import defaultdict, deque
from itertools import groupby
from math import hypot, sqrt
from pathlib import Path
from statistics import median

from generate import FIELDS, NOISE

COMMON = FIELDS[:3]
SUPPORTED = (*COMMON, "current_a")
SIGMA = dict(zip(FIELDS, NOISE))
CURRENT_SIGMA = hypot(SIGMA["current_a"], 0.12 * SIGMA["load_pct"])
PEER_SCALES = {field: sqrt(2) * SIGMA[field] for field in COMMON}
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


def detect_partition(path, machines, histories):
    for timestamp_text, rows in groupby(read_rows(path), key=lambda row: row["timestamp_ms"]):
        timestamp = int(timestamp_text)
        groups = defaultdict(list)
        for row in rows:
            observer = int(row["observer_id"])
            groups[machines[observer]].append((observer, row))
        for readings in groups.values():
            if len({observer for observer, _ in readings}) != 3 or len(readings) != 3:
                raise ValueError(f"Expected three distinct observers at {timestamp}")
            centers = {
                field: median(float(row[field]) for _, row in readings)
                for field in COMMON
            }
            expected_current = 2.5 + 0.12 * centers["load_pct"]
            for observer, row in readings:
                for field in COMMON:
                    score = abs(float(row[field]) - centers[field]) / PEER_SCALES[field]
                    if score > THRESHOLD:
                        yield (observer, timestamp, field), score, "peer_median"
                if row["current_a"]:
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

    parts = sorted((args.dataset / "observations").glob("part-*.csv"),
                   key=lambda path: int(path.stem.split("-")[1]))
    if not parts:
        parser.error("no observation partitions found")
    label_dir = args.dataset / "anomaly_labels"
    if args.evaluate and any(not (label_dir / part.name).is_file() for part in parts):
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
        for part in parts:
            labels = set()
            if args.evaluate:
                labels = {
                    (int(row["observer_id"]), int(row["timestamp_ms"]), row["field"])
                    for row in read_rows(label_dir / part.name)
                }
                all_labels += len(labels)
                supported_labels += sum(field in SUPPORTED for _, _, field in labels)
            for key, score, method in detect_partition(part, machines, histories):
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
