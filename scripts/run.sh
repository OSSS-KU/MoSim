#!/usr/bin/env bash
# Reproduce the cluster-level numbers on the artifact layout.
#
# From MoSim:
#   bash scripts/run.sh
#
# Outputs:
#   result/<single_job_profile>/runs/<inference_model>/<schedule>-timer.{txt,csv}
#   result/rawdata.csv
#   result/summary.csv
#   result/summary_by_run.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SIM_PY="simulator-trace-timer-bw.py"
MOSIM_BIN="mosim/target/release/mosim"
DATA_DIR="data"
RESULT_ROOT="${RESULT_ROOT:-result}"
TRACE_NAMES="${TRACE_NAMES:-32gpu 16gpu}"

# Set TRACE=1 to have every run also write a per-event trace log next to its
# other outputs. Off by default: ~224MB per run and ~2x slower.
TRACE="${TRACE:-0}"

BW_MBPS="${BW_MBPS:-463}"
INTRA_BW="${INTRA_BW:-16384}"
NUM_NODE="${NUM_NODE:-4}"
NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"
NUM_CPU_PER_NODE="${NUM_CPU_PER_NODE:-256}"

SINGLE_JOB_PROFILES="${SINGLE_JOB_PROFILES:-astrasim profiling testbed}"
SCHEDULES="${SCHEDULES:-colocate k8s-bin-packing k8s-load-balancing}"
INFERENCE_MODELS="${INFERENCE_MODELS:-none fixed fixed-10 mosim}"

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

need_file "$SIM_PY"
need_file "$MOSIM_BIN"

trace_csv_for() {
  case "$1" in
    32gpu|16gpu)
      echo "$DATA_DIR/trace/testbed-trace-$1.csv" ;;
    *)
      echo "ERROR: unknown trace '$1' (expected 32gpu or 16gpu)" >&2
      exit 1 ;;
  esac
}

testbed_dir_for() {
  case "$1" in
    32gpu|16gpu)
      echo "$DATA_DIR/testbed/$1" ;;
    *)
      echo "ERROR: unknown trace '$1' (expected 32gpu or 16gpu)" >&2
      exit 1 ;;
  esac
}

result_root_for_trace() {
  case "$1" in
    32gpu)
      echo "$RESULT_ROOT" ;;
    16gpu)
      echo "$RESULT_ROOT/16gpu" ;;
    *)
      echo "ERROR: unknown trace '$1' (expected 32gpu or 16gpu)" >&2
      exit 1 ;;
  esac
}

num_node_for_trace() {
  case "$1" in
    32gpu)
      echo "${NUM_NODE_32GPU:-$NUM_NODE}" ;;
    16gpu)
      echo "${NUM_NODE_16GPU:-2}" ;;
    *)
      echo "ERROR: unknown trace '$1' (expected 32gpu or 16gpu)" >&2
      exit 1 ;;
  esac
}

for trace_name in $TRACE_NAMES; do
  need_file "$(trace_csv_for "$trace_name")"
  case "$trace_name" in
    32gpu)
      need_file "$(testbed_dir_for "$trace_name")/colo.csv"
      need_file "$(testbed_dir_for "$trace_name")/k8sbp.csv"
      need_file "$(testbed_dir_for "$trace_name")/k8slb.csv"
      ;;
    16gpu)
      need_file "$(testbed_dir_for "$trace_name")/5sched_v4_allreduce_phy_result.csv"
      need_file "$(testbed_dir_for "$trace_name")/5sched_v4_allreduce_phy_perjobresult.csv"
      ;;
  esac
done

iter_csv_for_profile() {
  case "$1" in
    profiling)
      echo "$DATA_DIR/profiling/itertime_wshark-v100-8gpu-ar-loading-time.csv" ;;
    astrasim)
      echo "$DATA_DIR/single_job_simulation/itertime_simulated-ar-v100-8gpu-ar-loading-time.csv" ;;
    testbed)
      echo "" ;;
    *)
      echo "ERROR: unknown single_job_profile '$1'" >&2
      exit 1 ;;
  esac
}

net_csv_for_profile() {
  case "$1" in
    profiling)
      echo "$DATA_DIR/profiling/network_ar_v100_8gpu_network_summary.csv" ;;
    astrasim)
      echo "$DATA_DIR/single_job_simulation/network_chakra_ar_v100_8gpu_network_summary.csv" ;;
    testbed)
      echo "" ;;
    *)
      echo "ERROR: unknown single_job_profile '$1'" >&2
      exit 1 ;;
  esac
}

prepare_iteration_csv() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$dst" -nt "$src" && "${FORCE_PREPARE:-0}" != 1 ]]; then
    return 0
  fi
  python3 - "$src" "$dst" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
required = [
    "model_name",
    "gpu_workers",
    "iteration_computing_time",
    "iteration_networking_time",
    "loading_time",
    "colocated_model_name",
    "colocated_gpu_workers",
]
model_aliases = {
    "alexnet": "workload_01",
    "bert": "workload_02",
    "densenet100_k12": "workload_04",
    "densenet40_k12": "workload_05",
    "googlenet": "workload_06",
    "gpt2": "workload_07",
    "inception3": "workload_11",
    "resnet110": "workload_12",
    "resnet44": "workload_13",
    "resnet50": "workload_14",
    "vgg16": "workload_16",
    "wav2vec2": "workload_17",
    "whisper": "workload_18",
}

with src.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])

for name in required:
    if name not in fieldnames:
        fieldnames.append(name)

with dst.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        row["model_name"] = model_aliases.get(row.get("model_name", ""), row.get("model_name", ""))
        row["colocated_model_name"] = model_aliases.get(
            row.get("colocated_model_name", ""),
            row.get("colocated_model_name", ""),
        )
        if not (row.get("loading_time") or "").strip():
            row["loading_time"] = "0.0"
        row.setdefault("colocated_model_name", "")
        row.setdefault("colocated_gpu_workers", "")
        writer.writerow({name: row.get(name, "") for name in fieldnames})
PY
}

prepare_network_csv() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$dst" -nt "$src" && "${FORCE_PREPARE:-0}" != 1 ]]; then
    return 0
  fi
  python3 - "$src" "$dst" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model_aliases = {
    "alexnet": "workload_01",
    "bert": "workload_02",
    "densenet100_k12": "workload_04",
    "densenet40_k12": "workload_05",
    "googlenet": "workload_06",
    "gpt2": "workload_07",
    "inception3": "workload_11",
    "resnet110": "workload_12",
    "resnet44": "workload_13",
    "resnet50": "workload_14",
    "vgg16": "workload_16",
    "wav2vec2": "workload_17",
    "whisper": "workload_18",
}

with src.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])

if "Model" not in fieldnames:
    raise SystemExit(f"ERROR: missing Model column in {src}")

with dst.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        row["Model"] = model_aliases.get(row.get("Model", ""), row.get("Model", ""))
        writer.writerow({name: row.get(name, "") for name in fieldnames})
PY
}

prepare_trace_csv() {
  local src_trace="$1"
  local iter_csv="$2"
  local dst="$3"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$dst" -nt "$src_trace" && "$dst" -nt "$iter_csv" && "${FORCE_PREPARE:-0}" != 1 ]]; then
    return 0
  fi
  python3 - "$src_trace" "$iter_csv" "$dst" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

src_trace = Path(sys.argv[1])
iter_csv = Path(sys.argv[2])
dst = Path(sys.argv[3])

model_aliases = {
    "alexnet": "workload_01",
    "bert": "workload_02",
    "densenet100_k12": "workload_04",
    "densenet40_k12": "workload_05",
    "googlenet": "workload_06",
    "gpt2": "workload_07",
    "inception3": "workload_11",
    "resnet110": "workload_12",
    "resnet44": "workload_13",
    "resnet50": "workload_14",
    "vgg16": "workload_16",
    "wav2vec2": "workload_17",
    "whisper": "workload_18",
}


def to_float(value: str, *, field: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"ERROR: invalid {field}={value!r} in {context}")


def to_int(value: str, *, field: str, context: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise SystemExit(f"ERROR: invalid {field}={value!r} in {context}")


with iter_csv.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    iter_rows = list(reader)

iteration_time: dict[tuple[str, int], float] = {}
for lineno, row in enumerate(iter_rows, start=2):
    if (row.get("colocated_model_name") or "").strip() or (row.get("colocated_gpu_workers") or "").strip():
        continue
    model = (row.get("model_name") or "").strip()
    workers = to_int(row.get("gpu_workers", ""), field="gpu_workers", context=f"{iter_csv}:{lineno}")
    compute = to_float(
        row.get("iteration_computing_time", ""),
        field="iteration_computing_time",
        context=f"{iter_csv}:{lineno}",
    )
    networking = to_float(
        row.get("iteration_networking_time", ""),
        field="iteration_networking_time",
        context=f"{iter_csv}:{lineno}",
    )
    total = compute + networking
    if total <= 0.0:
        raise SystemExit(
            f"ERROR: non-positive iteration time for model={model}, gpu_workers={workers} "
            f"in {iter_csv}:{lineno}"
        )
    iteration_time[(model, workers)] = total

with src_trace.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])

required = ["model", "duration", "gpu_workers", "gpu_per_worker"]
for name in required:
    if name not in fieldnames:
        raise SystemExit(f"ERROR: trace {src_trace} is missing required column {name!r}")
if "num_iteration" not in fieldnames:
    fieldnames.append("num_iteration")

for lineno, row in enumerate(rows, start=2):
    model = model_aliases.get((row.get("model") or "").strip(), (row.get("model") or "").strip())
    row["model"] = model
    duration = to_float(row.get("duration", ""), field="duration", context=f"{src_trace}:{lineno}")
    gpu_workers = to_int(row.get("gpu_workers", ""), field="gpu_workers", context=f"{src_trace}:{lineno}")
    gpu_per_worker = to_int(
        row.get("gpu_per_worker", "1") or "1",
        field="gpu_per_worker",
        context=f"{src_trace}:{lineno}",
    )
    total_gpus = gpu_workers * gpu_per_worker
    per_iteration = iteration_time.get((model, total_gpus))
    if per_iteration is None:
        raise SystemExit(
            f"ERROR: no solo iteration profile for model={model}, total_gpus={total_gpus} "
            f"while rewriting {src_trace}:{lineno}"
        )
    row["num_iteration"] = f"{duration / per_iteration:.12g}"

import io

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
content = buf.getvalue()
if dst.exists() and dst.read_text(encoding="utf-8", errors="replace") == content:
    raise SystemExit(0)
dst.write_text(content, encoding="utf-8")
PY
}

run_one() {
  local single_job_profile="$1"
  local inference_model="$2"
  local schedule="$3"
  local profile_trace="$4"
  local net_csv="$5"
  local prepared="$6"
  local trace_name="$7"
  local trace_result_root="$8"
  local trace_num_node="$9"

  local out_dir="$trace_result_root/$single_job_profile/runs/$inference_model"
  mkdir -p "$out_dir"
  local log="$out_dir/${schedule}-timer.txt"
  local timer_csv="$out_dir/${schedule}-timer.csv"
  local alloc="$out_dir/${schedule}-allocation.log"
  local gpu_util="$out_dir/${schedule}-gpu_util.csv"
  local stdout_log="$out_dir/${schedule}-stdout.log"
  local stderr_log="$out_dir/${schedule}-stderr.log"
  local resolved_model=""
  local interference_ratio="none"

  local extra=()
  case "$inference_model" in
    none)
      resolved_model="none"
      extra=(--interference-model "$resolved_model") ;;
    fixed)
      resolved_model="fixed"
      interference_ratio="0.25"
      extra=(--interference-model "$resolved_model" --interference-ratio "$interference_ratio") ;;
    fixed-10)
      resolved_model="fixed"
      interference_ratio="0.10"
      extra=(--interference-model "$resolved_model" --interference-ratio "$interference_ratio") ;;
    mosim)
      resolved_model="mosim"
      extra=(--interference-model "$resolved_model") ;;
    *)
      echo "ERROR: unknown inference_model '$inference_model'" >&2
      exit 1 ;;
  esac

  # Appended after the case above, which assigns `extra` rather than appending.
  if [[ "$TRACE" == 1 ]]; then
    extra+=(--trace-log "$out_dir/${schedule}-trace.log")
  fi

  if [[ "${FORCE:-0}" != 1 && -f "$log" ]] \
      && grep -q '^avg_jct:' "$log" \
      && grep -q "^artifact_single_job_profile: $single_job_profile\$" "$log" \
      && grep -q "^artifact_inference_model: $inference_model\$" "$log" \
      && grep -q "^artifact_interference_model: $resolved_model\$" "$log" \
      && grep -q '^artifact_num_iterations: duration_div_profile_iteration_time$' "$log" \
      && grep -q '^artifact_output_mode: clean_before_run$' "$log"; then
    echo "skip: $trace_name / $single_job_profile / $inference_model / $schedule"
    return 0
  fi

  echo "run : $trace_name / $single_job_profile / $inference_model / $schedule"
  rm -f "$log" "$timer_csv" "$alloc" "$gpu_util" "$stdout_log" "$stderr_log"
  if ! MOSIM_BIN="$MOSIM_BIN" python3 "$SIM_PY" \
    --num_node "$trace_num_node" \
    --num_gpus_per_node "$NUM_GPU_PER_NODE" \
    --num_cpus_per_node "$NUM_CPU_PER_NODE" \
    --schedule "$schedule" \
    --jobtrace "$profile_trace" \
    --iteration_time_csv_file "$prepared" \
    --communication_volume_csv_file "$net_csv" \
    --bandwidth "$BW_MBPS" \
    --intra_bandwidth "$INTRA_BW" \
    --allocationlog "$alloc" \
    --log "$log" \
    --gpu_util_log "$gpu_util" \
    "${extra[@]}" >"$stdout_log" 2>"$stderr_log"; then
    echo "ERROR: simulation failed: $trace_name / $single_job_profile / $inference_model / $schedule" >&2
    echo "stdout: $stdout_log" >&2
    echo "stderr: $stderr_log" >&2
    tail -40 "$stderr_log" >&2 || true
    return 1
  fi

  need_file "$timer_csv"
  need_file "$log"
  {
    echo "artifact_single_job_profile: $single_job_profile"
    echo "artifact_inference_model: $inference_model"
    echo "artifact_interference_model: $resolved_model"
    echo "artifact_interference_ratio: $interference_ratio"
    echo "artifact_schedule: $schedule"
    echo "artifact_trace: $profile_trace"
    echo "artifact_trace_name: $trace_name"
    echo "artifact_num_node: $trace_num_node"
    echo "artifact_num_iterations: duration_div_profile_iteration_time"
    echo "artifact_output_mode: clean_before_run"
  } >> "$log"
}

summarize() {
  python3 - "$RESULT_ROOT" "$DATA_DIR" "$SINGLE_JOB_PROFILES" "$INFERENCE_MODELS" "$SCHEDULES" "$TRACE_NAMES" <<'PY'
from __future__ import annotations

import csv
import io
import math
import re
import statistics
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
single_job_profiles = sys.argv[3].split()
sim_profiles = [profile for profile in single_job_profiles if profile != "testbed"]
inference_models = sys.argv[4].split()
schedules = sys.argv[5].split()
trace_names = sys.argv[6].split()

metric_keys = [
    "avg_jct",
    "avg_waiting",
    "avg_training",
    "tail_jct",
    "tail_waiting",
    "tail_training",
    "Makespan",
]


def trace_result_root(trace: str) -> Path:
    return result_root if trace == "32gpu" else result_root / trace


def write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8", errors="replace") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_csv_if_changed(path: Path, fieldnames: list[str],
                         rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_if_changed(path, buf.getvalue())


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    k = (p / 100.0) * (len(values) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def mean(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return statistics.mean(values) if values else math.nan


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_testbed(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    jcts = [to_float(row.get("jcts", "")) for row in rows]
    waits = [to_float(row.get("wait_times", "")) for row in rows]
    trains = [to_float(row.get("training_time", "")) for row in rows]
    arrivals = [to_float(row.get("arrival_times", "")) for row in rows]
    ends = [to_float(row.get("end_time_seconds", "")) for row in rows]
    return {
        "avg_jct": mean(jcts),
        "avg_waiting": mean(waits),
        "avg_training": mean(trains),
        "tail_jct": percentile(jcts, 99.0),
        "tail_waiting": percentile(waits, 99.0),
        "tail_training": percentile(trains, 99.0),
        "Makespan": max(ends) - min(arrivals),
    }


def read_16gpu_testbed(path: Path) -> dict[str, dict[str, float]]:
    source_to_metric = {
        "AVERAGE JCT": "avg_jct",
        "AVERAGE WAITING TIME": "avg_waiting",
        "AVERAGE TRAINING TIME": "avg_training",
        "TAIL JCT": "tail_jct",
        "TAIL WAITING": "tail_waiting",
        "TAIL TRAINING": "tail_training",
        "MAKESPAN": "Makespan",
    }
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        schedule = (row.get("placement") or "").strip()
        if not schedule:
            continue
        out[schedule] = {
            metric: to_float(row.get(source, ""))
            for source, metric in source_to_metric.items()
        }
    return out


def read_testbed_for_trace(trace: str) -> dict[str, dict[str, float]]:
    if trace == "32gpu":
        sched_to_testbed = {
            "colocate": data_dir / "testbed/32gpu/colo.csv",
            "k8s-bin-packing": data_dir / "testbed/32gpu/k8sbp.csv",
            "k8s-load-balancing": data_dir / "testbed/32gpu/k8slb.csv",
        }
        return {
            sched: read_testbed(path)
            for sched, path in sched_to_testbed.items()
            if sched in schedules and path.is_file()
        }
    if trace == "16gpu":
        return {
            sched: metrics
            for sched, metrics in read_16gpu_testbed(
                data_dir / "testbed/16gpu/5sched_v4_allreduce_phy_result.csv"
            ).items()
            if sched in schedules
        }
    raise RuntimeError(f"unknown trace: {trace}")


def read_timer(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for key in metric_keys + ["sim_runtime_seconds", "cli_runtime_seconds"]:
        match = re.search(rf"^{re.escape(key)}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        if match:
            out[key] = float(match.group(1))
    if "avg_jct" not in out:
        raise RuntimeError(f"missing avg_jct in {path}")
    return out


def mape(sim: float, ref: float) -> float:
    if ref == 0 or not math.isfinite(ref):
        return math.nan
    return abs(sim - ref) / abs(ref) * 100.0


def schedule_from_testbed_path(path: Path) -> str:
    return {
        "colo.csv": "colocate",
        "k8sbp.csv": "k8s-bin-packing",
        "k8slb.csv": "k8s-load-balancing",
    }.get(path.name, path.stem)


def normalize_job_id(value: str) -> str:
    value = (value or "").strip()
    if value.lower().startswith("id") and value[2:].isdigit():
        return value[2:]
    return value


def read_testbed_raw_rows(path: Path, schedule: str,
                          timer_fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        col = {name: idx for idx, name in reversed(list(enumerate(header)))}
        source_for_timer = {
            "job_ids": "job_id",
            "arrival_times": "arrival_times",
            "start_times": "start_time_seconds",
            "end_times": "end_time_seconds",
            "wait_times": "wait_times",
            "queueing_wait_times": "queueing_wait_times",
            "capacity_wait_times": "capacity_wait_times",
            "placement_wait_times": "placement_wait_times",
            "training_times": "training_time",
            "jcts": "jcts",
        }
        rows: list[dict[str, str]] = []
        for values in reader:
            row = {field: "" for field in timer_fields}
            if "placement" in row:
                row["placement"] = schedule
            for timer_col, source_col in source_for_timer.items():
                if timer_col not in row or source_col not in col:
                    continue
                idx = col[source_col]
                value = values[idx] if idx < len(values) else ""
                row[timer_col] = normalize_job_id(value) if timer_col == "job_ids" else value
            rows.append(row)
        return rows


def format_float(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def read_16gpu_testbed_raw_rows(path: Path, schedule: str,
                                timer_fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for source in reader:
            if (source.get("placement") or "").strip() != schedule:
                continue
            row = {field: "" for field in timer_fields}
            if "placement" in row:
                row["placement"] = schedule
            source_for_timer = {
                "job_ids": "job_ids",
                "models": "models",
                "arrival_times": "arrival_times",
                "wait_times": "wait_times",
                "training_times": "training_time",
                "jcts": "jcts",
            }
            for timer_col, source_col in source_for_timer.items():
                if timer_col not in row:
                    continue
                value = source.get(source_col, "")
                row[timer_col] = normalize_job_id(value) if timer_col == "job_ids" else value

            arrival = to_float(source.get("arrival_times", ""))
            wait = to_float(source.get("wait_times", ""))
            jct = to_float(source.get("jcts", ""))
            if "start_times" in row:
                row["start_times"] = format_float(arrival + wait)
            if "end_times" in row:
                row["end_times"] = format_float(arrival + jct)
            rows.append(row)
    return rows


def read_testbed_raw_rows_for_trace(trace: str, schedule: str,
                                    timer_fields: list[str]) -> list[dict[str, str]]:
    if trace == "32gpu":
        path = {
            "colocate": data_dir / "testbed/32gpu/colo.csv",
            "k8s-bin-packing": data_dir / "testbed/32gpu/k8sbp.csv",
            "k8s-load-balancing": data_dir / "testbed/32gpu/k8slb.csv",
        }.get(schedule)
        return read_testbed_raw_rows(path, schedule, timer_fields) if path and path.is_file() else []
    if trace == "16gpu":
        return read_16gpu_testbed_raw_rows(
            data_dir / "testbed/16gpu/5sched_v4_allreduce_phy_perjobresult.csv",
            schedule,
            timer_fields,
        )
    raise RuntimeError(f"unknown trace: {trace}")


def write_testbed_outputs(trace: str, trace_root: Path,
                          testbed: dict[str, dict[str, float]],
                          timer_fields: list[str]) -> list[tuple[str, list[dict[str, str]]]]:
    out_dir = trace_root / "testbed/runs/testbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, list[dict[str, str]]]] = []
    for schedule in schedules:
        if schedule not in testbed:
            continue
        rows = read_testbed_raw_rows_for_trace(trace, schedule, timer_fields)
        if not rows:
            continue
        timer_csv = out_dir / f"{schedule}-timer.csv"
        write_csv_if_changed(timer_csv, timer_fields, rows)

        timer_txt = out_dir / f"{schedule}-timer.txt"
        text = "".join(f"{metric}: {testbed[schedule][metric]}\n" for metric in metric_keys)
        text += "artifact_single_job_profile: testbed\n"
        text += "artifact_inference_model: testbed\n"
        text += f"artifact_schedule: {schedule}\n"
        if trace != "32gpu":
            text += f"artifact_trace_name: {trace}\n"
        write_text_if_changed(timer_txt, text)
        out.append((schedule, rows))
    return out


def timer_paths_for_trace(trace: str) -> list[Path]:
    trace_root = trace_result_root(trace)
    timer_paths = sorted(trace_root.glob("*/runs/*/*-timer.csv"))
    return [
        path for path in timer_paths
        if path.relative_to(trace_root).parts[0] in sim_profiles
    ]


def write_rawdata() -> tuple[Path, int]:
    timer_paths_by_trace = {
        trace: timer_paths_for_trace(trace)
        for trace in trace_names
    }
    first_timer_path = next(
        (paths[0] for paths in timer_paths_by_trace.values() if paths),
        None,
    )
    if first_timer_path is None:
        raise RuntimeError(f"no timer CSV files found under {result_root}")

    with first_timer_path.open(newline="", encoding="utf-8-sig") as f:
        timer_fields = list(csv.DictReader(f).fieldnames or [])
    fields = ["single_job_profile", "inference_model", "schedule"] + timer_fields + ["trace"]
    rows: list[dict[str, str]] = []

    for trace in trace_names:
        trace_root = trace_result_root(trace)
        for path in timer_paths_by_trace[trace]:
            rel = path.relative_to(trace_root)
            single_job_profile = rel.parts[0]
            inference_model = rel.parts[2]
            schedule = path.name.removesuffix("-timer.csv")
            if inference_model not in inference_models or schedule not in schedules:
                continue
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for timer_row in reader:
                    row = {
                        "single_job_profile": single_job_profile,
                        "inference_model": inference_model,
                        "schedule": schedule,
                    }
                    row.update({field: timer_row.get(field, "") for field in timer_fields})
                    row["trace"] = trace
                    rows.append(row)

        testbed = testbed_by_trace.get(trace, {})
        for schedule, testbed_rows in write_testbed_outputs(trace, trace_root, testbed, timer_fields):
            for timer_row in testbed_rows:
                row = {
                    "single_job_profile": "testbed",
                    "inference_model": "testbed",
                    "schedule": schedule,
                }
                row.update({field: timer_row.get(field, "") for field in timer_fields})
                row["trace"] = trace
                rows.append(row)

    rawdata_csv = result_root / "rawdata.csv"
    write_csv_if_changed(rawdata_csv, fields, rows)
    return rawdata_csv, len(rows)


def build_summary_rows(trace: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trace_root = trace_result_root(trace)
    testbed = testbed_by_trace.get(trace, {})
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    for single_job_profile in sim_profiles:
        for inference_model in inference_models:
            for sched in schedules:
                if sched not in testbed:
                    continue
                timer_path = trace_root / single_job_profile / "runs" / inference_model / f"{sched}-timer.txt"
                if not timer_path.is_file():
                    continue
                sim = read_timer(timer_path)
                wide = {
                    "single_job_profile": single_job_profile,
                    "inference_model": inference_model,
                    "schedule": sched,
                    "sim_runtime_seconds": sim.get("sim_runtime_seconds", math.nan),
                    "cli_runtime_seconds": sim.get("cli_runtime_seconds", math.nan),
                }
                for metric in metric_keys:
                    ref = testbed[sched][metric]
                    sim_value = sim[metric]
                    err = mape(sim_value, ref)
                    long_rows.append({
                        "single_job_profile": single_job_profile,
                        "inference_model": inference_model,
                        "schedule": sched,
                        "metric": metric,
                        "sim": f"{sim_value:.6f}",
                        "testbed": f"{ref:.6f}",
                        "mape_pct": f"{err:.6f}",
                    })
                    wide[f"sim_{metric}"] = f"{sim_value:.6f}"
                    wide[f"mape_{metric}_pct"] = f"{err:.6f}"
                wide_rows.append(wide)
    return long_rows, wide_rows


def write_summary(trace: str,
                  long_rows: list[dict[str, object]],
                  wide_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    trace_root = trace_result_root(trace)
    summary_csv = trace_root / "summary.csv"
    summary_wide_csv = trace_root / "summary_by_run.csv"
    fields = ["single_job_profile", "inference_model", "schedule", "metric", "sim", "testbed", "mape_pct"]
    write_csv_if_changed(summary_csv, fields, long_rows)

    wide_fields = [
        "single_job_profile",
        "inference_model",
        "schedule",
        "sim_runtime_seconds",
        "cli_runtime_seconds",
    ]
    for metric in metric_keys:
        wide_fields.extend([f"sim_{metric}", f"mape_{metric}_pct"])
    write_csv_if_changed(summary_wide_csv, wide_fields, wide_rows)
    return summary_csv, summary_wide_csv


testbed_by_trace = {trace: read_testbed_for_trace(trace) for trace in trace_names}
summary_outputs: list[tuple[str, Path, Path, list[dict[str, object]]]] = []
for trace in trace_names:
    long_rows, wide_rows = build_summary_rows(trace)
    summary_csv, summary_wide_csv = write_summary(trace, long_rows, wide_rows)
    summary_outputs.append((trace, summary_csv, summary_wide_csv, wide_rows))

rawdata_csv, rawdata_rows = write_rawdata()

print()
print(f"wrote {rawdata_csv} ({rawdata_rows} rows)")
for trace, summary_csv, summary_wide_csv, _ in summary_outputs:
    print(f"wrote {summary_csv} ({trace})")
    print(f"wrote {summary_wide_csv} ({trace})")
print()
print("trace,single_job_profile,inference_model,schedule,avg_jct_mape,p99_jct_mape,makespan_mape")
for trace, _, _, wide_rows in summary_outputs:
    for row in wide_rows:
        print(
            f"{trace},{row['single_job_profile']},{row['inference_model']},{row['schedule']},"
            f"{row['mape_avg_jct_pct']},{row['mape_tail_jct_pct']},"
            f"{row['mape_Makespan_pct']}"
        )
PY
}

echo "root       : $ROOT"
echo "traces     : $TRACE_NAMES"
echo "result     : $RESULT_ROOT"
echo "bandwidth  : $BW_MBPS MB/s"
echo

for trace_name in $TRACE_NAMES; do
  trace_csv="$(trace_csv_for "$trace_name")"
  trace_result_root="$(result_root_for_trace "$trace_name")"
  trace_num_node="$(num_node_for_trace "$trace_name")"
  echo "trace      : $trace_name ($trace_csv, nodes=$trace_num_node, result=$trace_result_root)"
  echo

  for single_job_profile in $SINGLE_JOB_PROFILES; do
    if [[ "$single_job_profile" == "testbed" ]]; then
      continue
    fi

    iter_csv="$(iter_csv_for_profile "$single_job_profile")"
    net_csv="$(net_csv_for_profile "$single_job_profile")"
    need_file "$iter_csv"
    need_file "$net_csv"
    prepared="$trace_result_root/$single_job_profile/_build/$(basename "${iter_csv%.csv}")_prepared.csv"
    prepared_net="$trace_result_root/$single_job_profile/_build/$(basename "${net_csv%.csv}")_prepared.csv"
    profile_trace="$trace_result_root/$single_job_profile/_build/testbed-trace_duration-derived.csv"
    prepare_iteration_csv "$iter_csv" "$prepared"
    prepare_network_csv "$net_csv" "$prepared_net"
    prepare_trace_csv "$trace_csv" "$prepared" "$profile_trace"
    for inference_model in $INFERENCE_MODELS; do
      for schedule in $SCHEDULES; do
        run_one \
          "$single_job_profile" \
          "$inference_model" \
          "$schedule" \
          "$profile_trace" \
          "$prepared_net" \
          "$prepared" \
          "$trace_name" \
          "$trace_result_root" \
          "$trace_num_node"
      done
    done
  done
done

summarize
