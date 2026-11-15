"""Reproducible OFFLINE evaluation of injected synthetic operating anomalies.

This evaluates detection of artificially perturbed plant telemetry, NOT actual
electrical faults or hardware-level performance. Requires scikit-learn.
Run: python validation/validate.py --samples 4000 --seed 42
"""
import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "Gas Plant": ["power_output", "demand", "fuel_consumption", "emissions"],
    "Wind Farm": ["power_output", "demand", "wind_speed", "turbine_efficiency"],
    "Solar Farm": ["power_output", "demand", "solar_radiation", "panel_temperature"],
    "Hydroelectric Plant": ["power_output", "demand", "water_flow_rate", "turbine_rotation_speed"],
}


def load_generator():
    path = ROOT / "producer" / "producer-3.py"
    spec = importlib.util.spec_from_file_location("energy_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_data


def scores(truth, predicted):
    tp = sum(y == 1 and p == 1 for y, p in zip(truth, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(truth, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(truth, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(truth, predicted))
    n = tp + fp + tn + fn
    div = lambda a, b: round(a / b, 6) if b else None
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "samples": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": div(tp + tn, n),
        "precision": precision, "recall": recall,
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": div(fp, fp + tn),
        "injection_rate": div(tp + fn, n),
        "predicted_outlier_rate": div(tp + fp, n),
    }


def evaluate(samples=4000, seed=42, train_fraction=0.7, window=500):
    if samples < 400 or not 0 < train_fraction < 1 or window < 50:
        raise ValueError("Use at least 400 samples, 0 < train_fraction < 1 and window >= 50")
    random.seed(seed)
    generate_data = load_generator()
    by_plant = defaultdict(list)
    for index in range(samples):
        record = generate_data(index)
        by_plant[record["plant_type"]].append(record)

    result = {
        "scope": "synthetic injected operational anomalies; not verified electrical equipment faults",
        "seed": seed, "generated_samples": samples, "train_fraction": train_fraction,
        "window": window, "contamination": 0.05, "split": "chronological per plant",
        "per_plant": {},
    }
    all_true, all_pred = [], []
    for plant, features in FEATURES.items():
        rows = by_plant[plant]
        split = int(len(rows) * train_fraction)
        train, test = rows[:split], rows[split:]
        if len(train) < 50 or not test:
            raise ValueError(f"Insufficient train/test observations for {plant}; increase --samples")
        # Mirror the application's per-plant feature set and 5% contamination.
        model = IsolationForest(contamination=0.05, random_state=seed)
        model.fit([[r[k] for k in features] for r in train[-window:]])
        truth = [int(r["is_anomaly"]) for r in test]
        predicted = [int(label == -1) for label in model.predict(
            [[r[k] for k in features] for r in test]
        )]
        entry = scores(truth, predicted)
        entry["train_samples"] = len(train)
        result["per_plant"][plant] = entry
        all_true.extend(truth)
        all_pred.extend(predicted)
    result["overall"] = scores(all_true, all_pred)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--output", type=Path, help="Optional path for JSON report")
    args = parser.parse_args()
    report = evaluate(samples=args.samples, seed=args.seed, window=args.window)
    formatted = json.dumps(report, indent=2, sort_keys=True)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
