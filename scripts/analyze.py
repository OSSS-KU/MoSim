#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MethodSpec:
    label: str
    single_job_profile: str
    inference_model: str


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str


TIRESIAS = MethodSpec("Tiresias", "profiling", "none")
POLLUX = MethodSpec("Pollux", "profiling", "fixed-10")
MOSIM = MethodSpec("MoSim", "astrasim", "mosim")

TABLE2_METHODS = [TIRESIAS, POLLUX]
PAPER_METHODS = [TIRESIAS, POLLUX, MOSIM]

TABLE2_TRACE = "16gpu"
TABLE2_SCHEDULE = "k8s-load-balancing"

TABLE3_TRACE = "32gpu"
TABLE_SCHEDULES = [
    ("k8s-bin-packing", "Bin packing"),
    ("k8s-load-balancing", "Load balancing"),
]

TABLE2_METRICS = [
    MetricSpec("avg_training", "Average training time"),
    MetricSpec("avg_jct", "Average JCT"),
    MetricSpec("p99_jct", "P99 JCT"),
    MetricSpec("makespan", "Makespan"),
]

TABLE3_METRICS = [
    MetricSpec("avg_training", "Average training time"),
    MetricSpec("avg_jct", "Average JCT"),
    MetricSpec("median_jct", "Median JCT"),
    MetricSpec("p99_jct", "P99 JCT"),
    MetricSpec("makespan", "Makespan"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the paper Table II, Table III, and Table IV artifacts "
            "from one rawdata.csv file."
        )
    )
    parser.add_argument(
        "--rawdata",
        default="result/rawdata.csv",
        help="rawdata.csv emitted by scripts/run.sh",
    )
    parser.add_argument(
        "--outdir",
        default="result/paper",
        help="output root; tab2/tab3/tab4 are created below it",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="decimal places for table values",
    )
    return parser.parse_args()


def read_rawdata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing rawdata CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"ERROR: rawdata CSV has no header: {path}")
    return fieldnames, rows


def to_float(value: str | None) -> float:
    try:
        return float(value or "")
    except ValueError:
        return math.nan


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def percentile(values: list[float], p: float) -> float:
    values = sorted(finite(values))
    if not values:
        return math.nan
    k = (p / 100.0) * (len(values) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def mape(sim: float, ref: float) -> float:
    if not math.isfinite(sim) or not math.isfinite(ref):
        return math.nan
    if ref == 0:
        return 0.0 if sim == 0 else math.nan
    return abs(sim - ref) / abs(ref) * 100.0


def normalize_job_id(value: str | None) -> str:
    value = (value or "").strip()
    if value.lower().startswith("id") and value[2:].isdigit():
        return value[2:]
    return value


def job_sort_key(row: dict[str, str]) -> tuple[int, str]:
    job_id = normalize_job_id(row.get("job_ids"))
    return (int(job_id), job_id) if job_id.isdigit() else (10**9, job_id)


def select_rows(
    rows: Iterable[dict[str, str]],
    *,
    trace: str,
    schedule: str,
    single_job_profile: str,
    inference_model: str,
) -> list[dict[str, str]]:
    return sorted(
        [
            row
            for row in rows
            if (row.get("trace") or "32gpu") == trace
            and row.get("schedule") == schedule
            and row.get("single_job_profile") == single_job_profile
            and row.get("inference_model") == inference_model
        ],
        key=job_sort_key,
    )


def require_rows(rows: list[dict[str, str]], description: str) -> None:
    if not rows:
        raise SystemExit(f"ERROR: no rawdata rows found for {description}")


def job_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    training = finite(to_float(row.get("training_times")) for row in rows)
    jcts = finite(to_float(row.get("jcts")) for row in rows)
    end_times = finite(to_float(row.get("end_times")) for row in rows)
    if not training or not jcts or not end_times:
        raise SystemExit("ERROR: selected rawdata rows are missing required metric values")
    return {
        "avg_training": statistics.mean(training),
        "avg_jct": statistics.mean(jcts),
        "median_jct": statistics.median(jcts),
        "p99_jct": percentile(jcts, 99.0),
        "makespan": max(end_times),
    }


def jct_values(rows: list[dict[str, str]]) -> list[float]:
    values = finite(to_float(row.get("jcts")) for row in rows)
    if not values:
        raise SystemExit("ERROR: selected rawdata rows are missing jcts")
    return values


def ks_distance(sample_a: list[float], sample_b: list[float]) -> float:
    a = sorted(finite(sample_a))
    b = sorted(finite(sample_b))
    if not a or not b:
        return math.nan
    i = 0
    j = 0
    max_gap = 0.0
    values = sorted(set(a + b))
    for value in values:
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        max_gap = max(max_gap, abs(i / len(a) - j / len(b)))
    return max_gap


def wasserstein_distance(sample_a: list[float], sample_b: list[float]) -> float:
    a = sorted(finite(sample_a))
    b = sorted(finite(sample_b))
    if not a or not b:
        return math.nan
    if len(a) == len(b):
        return statistics.mean(abs(x - y) for x, y in zip(a, b))

    # Fallback for unequal sample sizes: compare quantiles on a dense grid.
    grid_size = max(len(a), len(b))
    total = 0.0
    for idx in range(grid_size):
        q = idx / (grid_size - 1) if grid_size > 1 else 0.0
        total += abs(quantile(a, q) - quantile(b, q))
    return total / grid_size


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    q = min(max(q, 0.0), 1.0)
    k = q * (len(values) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def format_number(value: float, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_raw_subset(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.get("trace", ""),
            row.get("schedule", ""),
            row.get("single_job_profile", ""),
            row.get("inference_model", ""),
            job_sort_key(row),
        ),
    )
    write_csv(path, fieldnames, ordered_rows)


def write_markdown(path: Path, title: str, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table2(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    outdir: Path,
    precision: int,
) -> None:
    tab = outdir / "tab2"
    schedule = TABLE2_SCHEDULE
    testbed = select_rows(
        rows,
        trace=TABLE2_TRACE,
        schedule=schedule,
        single_job_profile="testbed",
        inference_model="testbed",
    )
    require_rows(testbed, f"Table II testbed {TABLE2_TRACE}/{schedule}")
    testbed_metrics = job_metrics(testbed)

    subset = list(testbed)
    values: dict[str, dict[str, float]] = {}
    metric_rows: list[dict[str, object]] = []
    for method in TABLE2_METHODS:
        selected = select_rows(
            rows,
            trace=TABLE2_TRACE,
            schedule=schedule,
            single_job_profile=method.single_job_profile,
            inference_model=method.inference_model,
        )
        require_rows(selected, f"Table II {method.label} {TABLE2_TRACE}/{schedule}")
        subset.extend(selected)
        sim_metrics = job_metrics(selected)
        values[method.label] = {
            metric.key: mape(sim_metrics[metric.key], testbed_metrics[metric.key])
            for metric in TABLE2_METRICS
        }
        for metric in TABLE2_METRICS:
            metric_rows.append(
                {
                    "trace": TABLE2_TRACE,
                    "schedule": schedule,
                    "method": method.label,
                    "single_job_profile": method.single_job_profile,
                    "inference_model": method.inference_model,
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "sim_value": f"{sim_metrics[metric.key]:.6f}",
                    "testbed_value": f"{testbed_metrics[metric.key]:.6f}",
                    "mape_pct": f"{values[method.label][metric.key]:.6f}",
                }
            )

    write_raw_subset(tab / "data" / "rawdata_subset.csv", fieldnames, subset)
    write_csv(
        tab / "data" / "metrics.csv",
        [
            "trace",
            "schedule",
            "method",
            "single_job_profile",
            "inference_model",
            "metric",
            "metric_label",
            "sim_value",
            "testbed_value",
            "mape_pct",
        ],
        metric_rows,
    )

    md_rows = [
        [
            metric.label,
            format_number(values[TIRESIAS.label][metric.key], precision),
            format_number(values[POLLUX.label][metric.key], precision),
        ]
        for metric in TABLE2_METRICS
    ]
    write_markdown(
        tab / "out" / "table2.md",
        "Table II: Simulation Fidelity (MAPE %) of Existing Simulators",
        ["Metric", "Tiresias (No-contention)", "Pollux (Static-contention)"],
        md_rows,
    )
    write_csv(
        tab / "out" / "table2.csv",
        ["metric", "Tiresias (No-contention)", "Pollux (Static-contention)"],
        [
            {
                "metric": metric.label,
                "Tiresias (No-contention)": format_number(values[TIRESIAS.label][metric.key], precision),
                "Pollux (Static-contention)": format_number(values[POLLUX.label][metric.key], precision),
            }
            for metric in TABLE2_METRICS
        ],
    )


def build_table3(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    outdir: Path,
    precision: int,
) -> None:
    tab = outdir / "tab3"
    subset: list[dict[str, str]] = []
    values: dict[tuple[str, str], dict[str, float]] = {}
    metric_rows: list[dict[str, object]] = []

    for schedule, _schedule_label in TABLE_SCHEDULES:
        testbed = select_rows(
            rows,
            trace=TABLE3_TRACE,
            schedule=schedule,
            single_job_profile="testbed",
            inference_model="testbed",
        )
        require_rows(testbed, f"Table III testbed {TABLE3_TRACE}/{schedule}")
        subset.extend(testbed)
        testbed_metrics = job_metrics(testbed)

        for method in PAPER_METHODS:
            selected = select_rows(
                rows,
                trace=TABLE3_TRACE,
                schedule=schedule,
                single_job_profile=method.single_job_profile,
                inference_model=method.inference_model,
            )
            require_rows(selected, f"Table III {method.label} {TABLE3_TRACE}/{schedule}")
            subset.extend(selected)
            sim_metrics = job_metrics(selected)
            values[(schedule, method.label)] = {
                metric.key: mape(sim_metrics[metric.key], testbed_metrics[metric.key])
                for metric in TABLE3_METRICS
            }
            for metric in TABLE3_METRICS:
                metric_rows.append(
                    {
                        "trace": TABLE3_TRACE,
                        "schedule": schedule,
                        "method": method.label,
                        "single_job_profile": method.single_job_profile,
                        "inference_model": method.inference_model,
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "sim_value": f"{sim_metrics[metric.key]:.6f}",
                        "testbed_value": f"{testbed_metrics[metric.key]:.6f}",
                        "mape_pct": f"{values[(schedule, method.label)][metric.key]:.6f}",
                    }
                )

    write_raw_subset(tab / "data" / "rawdata_subset.csv", fieldnames, subset)
    write_csv(
        tab / "data" / "metrics.csv",
        [
            "trace",
            "schedule",
            "method",
            "single_job_profile",
            "inference_model",
            "metric",
            "metric_label",
            "sim_value",
            "testbed_value",
            "mape_pct",
        ],
        metric_rows,
    )

    header = ["Method"]
    for _, schedule_label in TABLE_SCHEDULES:
        header.extend(f"{schedule_label} {metric.label}" for metric in TABLE3_METRICS)
    md_rows = []
    csv_rows = []
    for method in PAPER_METHODS:
        md_row = [method.label]
        csv_row: dict[str, object] = {"Method": method.label}
        for schedule, schedule_label in TABLE_SCHEDULES:
            for metric in TABLE3_METRICS:
                formatted = format_number(values[(schedule, method.label)][metric.key], precision)
                md_row.append(formatted)
                csv_row[f"{schedule_label} {metric.label}"] = formatted
        md_rows.append(md_row)
        csv_rows.append(csv_row)

    write_markdown(
        tab / "out" / "table3.md",
        "Table III: Simulation Fidelity (MAPE %)",
        header,
        md_rows,
    )
    write_csv(tab / "out" / "table3.csv", header, csv_rows)


def build_table4(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    outdir: Path,
    precision: int,
) -> None:
    tab = outdir / "tab4"
    subset: list[dict[str, str]] = []
    values: dict[tuple[str, str], dict[str, float]] = {}
    distance_rows: list[dict[str, object]] = []

    for schedule, _schedule_label in TABLE_SCHEDULES:
        testbed = select_rows(
            rows,
            trace=TABLE3_TRACE,
            schedule=schedule,
            single_job_profile="testbed",
            inference_model="testbed",
        )
        require_rows(testbed, f"Table IV testbed {TABLE3_TRACE}/{schedule}")
        subset.extend(testbed)
        testbed_jcts = jct_values(testbed)

        for method in PAPER_METHODS:
            selected = select_rows(
                rows,
                trace=TABLE3_TRACE,
                schedule=schedule,
                single_job_profile=method.single_job_profile,
                inference_model=method.inference_model,
            )
            require_rows(selected, f"Table IV {method.label} {TABLE3_TRACE}/{schedule}")
            subset.extend(selected)
            sim_jcts = jct_values(selected)
            ks = ks_distance(sim_jcts, testbed_jcts)
            wasserstein = wasserstein_distance(sim_jcts, testbed_jcts)
            values[(schedule, method.label)] = {
                "ks_distance": ks,
                "wasserstein": wasserstein,
            }
            distance_rows.append(
                {
                    "trace": TABLE3_TRACE,
                    "schedule": schedule,
                    "method": method.label,
                    "single_job_profile": method.single_job_profile,
                    "inference_model": method.inference_model,
                    "ks_distance": f"{ks:.6f}",
                    "wasserstein_seconds": f"{wasserstein:.6f}",
                }
            )

    write_raw_subset(tab / "data" / "rawdata_subset.csv", fieldnames, subset)
    write_csv(
        tab / "data" / "distances.csv",
        [
            "trace",
            "schedule",
            "method",
            "single_job_profile",
            "inference_model",
            "ks_distance",
            "wasserstein_seconds",
        ],
        distance_rows,
    )

    header = [
        "Method",
        "Bin packing KS distance",
        "Bin packing Wasserstein (s)",
        "Load balancing KS distance",
        "Load balancing Wasserstein (s)",
    ]
    md_rows = []
    csv_rows = []
    for method in PAPER_METHODS:
        md_row = [method.label]
        csv_row: dict[str, object] = {"Method": method.label}
        for schedule, schedule_label in TABLE_SCHEDULES:
            ks = format_number(values[(schedule, method.label)]["ks_distance"], precision)
            wasserstein = format_number(values[(schedule, method.label)]["wasserstein"], 0)
            md_row.extend([ks, wasserstein])
            csv_row[f"{schedule_label} KS distance"] = ks
            csv_row[f"{schedule_label} Wasserstein (s)"] = wasserstein
        md_rows.append(md_row)
        csv_rows.append(csv_row)

    write_markdown(
        tab / "out" / "table4.md",
        "Table IV: Distribution-Level Fidelity",
        header,
        md_rows,
    )
    write_csv(tab / "out" / "table4.csv", header, csv_rows)


def main() -> None:
    args = parse_args()
    rawdata = Path(args.rawdata)
    outdir = Path(args.outdir)
    fieldnames, rows = read_rawdata(rawdata)

    build_table2(rows, fieldnames, outdir, args.precision)
    build_table3(rows, fieldnames, outdir, args.precision)
    build_table4(rows, fieldnames, outdir, args.precision)

    print(f"wrote {outdir / 'tab2'}")
    print(f"wrote {outdir / 'tab3'}")
    print(f"wrote {outdir / 'tab4'}")


if __name__ == "__main__":
    main()
