#!/usr/bin/env python3
"""User-facing CLI for the MoSim GPU cluster placement simulator.

This script is a thin front-end. It is responsible only for:
  1. Parsing and validating command-line arguments via argparse.
  2. Forwarding the validated config as JSON over stdin to the Rust
     simulator binary (mosim), which contains all simulation logic.

The Rust binary writes all output files (allocation log, timer log,
CSV) and prints all metrics. This script just streams its stdout/stderr.

Example (after running `bash scripts/run.sh` to prepare the inputs):
    python simulator-trace-timer-bw.py --num_node 4 --num_gpus_per_node 8 \
        --num_cpus_per_node 256 --schedule k8s-bin-packing \
        --jobtrace result/astrasim/_build/testbed-trace_duration-derived.csv \
        --iteration_time_csv_file \
            result/astrasim/_build/itertime_simulated-ar-v100-8gpu-ar-loading-time_prepared.csv \
        --communication_volume_csv_file \
            result/astrasim/_build/network_chakra_ar_v100_8gpu_network_summary_prepared.csv \
        --allocationlog result/example-allocation.log \
        --log result/example-timer.txt --bandwidth 463

Override the binary location with MOSIM_BIN if needed:
    MOSIM_BIN=/path/to/mosim python simulator-trace-timer-bw.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCHEDULE_CHOICES = [
    "k8s-load-balancing",
    "k8s-bin-packing",
    "colocate",
    "tiresias",
]

INTERFERENCE_CHOICES = [
    # `mosim` is the configuration used in the MASCOTS 2026 paper. It is an
    # alias for `comms-iter-intra`, which names the mechanism instead.
    "mosim",
    "none",
    "fixed",
    "corun-profile",
    "comms",
    "comms-iter",
    "comms-iter-intra",
    "comms-iter-lcm",
    "comms-iter-intra-lcm",
]

# Which interference models actually read each optional knob.
#
# Measured, not inferred: every flag below was added to an otherwise identical
# 60-job / 32-GPU run and the resulting makespan compared. A model is listed
# only where the flag changed the result. Passing a flag to a model that is not
# listed parses fine and changes nothing, so `warn_unused_flags` says so out
# loud rather than letting a sensitivity study quietly measure nothing.
_INTRA = ("mosim", "comms-iter-intra", "comms-iter-intra-lcm")
_ITER = _INTRA + ("comms-iter", "comms-iter-lcm")
_COMMS = _ITER + ("comms",)
_LCM = ("comms-iter-lcm", "comms-iter-intra-lcm")

_INTRA_NAME = "the intra models (mosim, comms-iter-intra-lcm)"
_ITER_NAME = "the iteration-level models (mosim, comms-iter, comms-iter*-lcm)"
_COMMS_NAME = "comms and the comms-iter* family"
_LCM_NAME = "the LCM models (comms-iter-lcm, comms-iter-intra-lcm)"

# (flag, argparse dest, default value, models that read it, human-readable group)
FLAG_READERS = [
    ("--contention_model",              "contention_model",              None,  _INTRA, _INTRA_NAME),
    ("--mm1_epsilon",                   "mm1_epsilon",                   None,  _INTRA, _INTRA_NAME),
    ("--comm_contention_fraction",      "comm_contention_fraction",      None,  _INTRA, _INTRA_NAME),
    ("--phase_overlap_weighted",        "phase_overlap_weighted",        False, _ITER,  _ITER_NAME),
    ("--hol_efficiency",                "hol_efficiency",                None,  _COMMS, _COMMS_NAME),
    ("--min_guaranteed_bw",             "min_guaranteed_bw",             None,  _COMMS, _COMMS_NAME),
    ("--required_bandwidth_cap_factor", "required_bandwidth_cap_factor",  None,  _COMMS, _COMMS_NAME),
    ("--cap_n_exponent",                "cap_n_exponent",                None,  _COMMS, _COMMS_NAME),
    ("--lcm-time-decimals",             "lcm_time_decimals",             6,     _LCM,   _LCM_NAME),
    ("--disable-lcm-cycle-jump",        "enable_lcm_cycle_jump",          True,  _LCM,   _LCM_NAME),
]


class _RemovedFlagAction(argparse.Action):
    """Emit a clear error when a removed flag is used."""

    def __init__(self, option_strings, dest, message, **kwargs):
        kwargs.setdefault("nargs", "?")
        kwargs.setdefault("default", argparse.SUPPRESS)
        super().__init__(option_strings, dest, **kwargs)
        self._message = message

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(self._message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MoSim GPU cluster placement simulator')
    parser.add_argument('--num_node', '-n', default=8, type=int,
                        help='number of servers')
    parser.add_argument('--num_gpus_per_node', '-g', default=8, type=int,
                        help='number of GPUs per server')
    parser.add_argument('--num_cpus_per_node', '-c', default=256, type=int,
                        help='number of CPU cores per server')
    parser.add_argument('--schedule', '-s', required=True,
                        choices=SCHEDULE_CHOICES,
                        help='Placement (a.k.a. schedule) algorithm. '
                             'FIFO queueing is implicit for all choices.')
    # Backward-compat trap: --placement was merged into --schedule.
    parser.add_argument('--placement', '-p',
                        action=_RemovedFlagAction,
                        message="--placement has been removed; "
                                "use --schedule with the same value instead.")
    parser.add_argument('--jobtrace', '-t', required=True,
                        help='Prepared job trace CSV. scripts/run.sh writes examples '
                             'under result/<profile>/_build/.')
    parser.add_argument('--allocationlog', '-a', default='result/allocation_log.txt')
    parser.add_argument('--log', '-l', default='result/timer-result.txt')
    parser.add_argument('--gpu_util_log', default='',
                        help='Optional CSV path for per-event GPU utilization log. '
                             'Columns: timestamp_sec, cluster_avg_util, cluster_total_util, '
                             'cluster_active_gpus, node0_avg_util, ..., node{N-1}_avg_util. '
                             'A row is emitted whenever GPU allocation state changes '
                             '(job placement / completion). Empty string disables logging.')
    parser.add_argument('--trace-log', dest='trace_log', default='',
                        help='Optional path for the per-event trace log (scheduling '
                             'decisions, phase transitions, contention recomputations). '
                             'Empty string (the default) disables it: the trace is '
                             '~224MB per 60-job run and writing it costs about half the '
                             'runtime. stdout always carries the final metrics summary.')
    parser.add_argument('--bandwidth', '-b', default=375, type=int,
                        help='inter-server interconnect bandwidth, unit=MB/s (1Gbps = 125MB/s)')
    parser.add_argument('--intra_bandwidth', '-ib', default=16384, type=int,
                        help='intra-server interconnect bandwidth, unit=MB/s (16GB/s = 16384MB/s)')
    parser.add_argument('--iteration_time_csv_file', '-it', required=True,
                        help='Per-(model, gpu_workers) iteration compute/networking '
                             'time + loading time, with optional colocated profiling rows. '
                             'scripts/run.sh writes prepared examples under '
                             'result/<profile>/_build/.')
    parser.add_argument('--communication_volume_csv_file', '-cv', required=True,
                        help='Network summary CSV; the (Model, Number of Workers) row '
                             'provides "Sum of Max TX+RX (MB/s)" used as profiled_network. '
                             'scripts/run.sh writes prepared examples under '
                             'result/<profile>/_build/.')
    # Backward-compat trap: loading time now comes from --iteration_time_csv_file.
    parser.add_argument('--loading_time_csv_file', '-lt',
                        action=_RemovedFlagAction,
                        message="--loading_time_csv_file has been removed; "
                                "loading_time is now read from --iteration_time_csv_file.")
    parser.add_argument('--interference-model', dest='interference_model',
                        default='mosim',
                        choices=INTERFERENCE_CHOICES,
                        help='Interference model. `mosim` (default) is the paper configuration, an alias for `comms-iter-intra`. '
                             'Note `comms` is a different, job-level model and does NOT reproduce the paper. '
                             'fixed requires --interference-ratio. '
                             'corun-profile reads compute/networking times from the '
                             'iteration_time CSV (including colocated rows) and is '
                             'incompatible with --interference-ratio and LCM options.')
    parser.add_argument('--interference-ratio', dest='interference_ratio',
                        type=float, default=None,
                        help='Slowdown for fixed model (e.g., 0.1 = 1.1x slowdown).')
    parser.add_argument('--lcm-algo', dest='lcm_algo', default='gcd_fold',
                        choices=['gcd_fold', 'binary_gcd_fold', 'prime_factor_merge',
                                 'spf_factor_merge', 'reduce_tree'],
                        help='LCM algorithm (only for comms-iter-lcm and comms-iter-intra-lcm).')
    parser.add_argument('--lcm-time-decimals', dest='lcm_time_decimals',
                        type=int, default=6,
                        help='Decimal precision for LCM tick scaling. Uses round(x * 10^N).')
    parser.set_defaults(enable_lcm_cycle_jump=True)
    parser.add_argument('--enable-lcm-cycle-jump', dest='enable_lcm_cycle_jump',
                        action='store_true',
                        help='Enable opportunistic cycle jump for comms-iter-lcm variants (default).')
    parser.add_argument('--disable-lcm-cycle-jump', dest='enable_lcm_cycle_jump',
                        action='store_false',
                        help='Disable opportunistic cycle jump for comms-iter-lcm variants.')
    parser.add_argument('--overlapping_ratio', type=float, default=0.0,
                        help='Fraction of compute time (0..1) that may overlap with communication; '
                             '0.0 preserves legacy behavior (no overlap).')
    parser.add_argument('--required_bandwidth_cap_factor', type=float, default=None,
                        help='Cap each job\'s per-server required_bandwidth at '
                             'this multiple of --bandwidth (server NIC). Default '
                             'None preserves legacy unclamped behaviour. e.g. '
                             '1.4 means R is clamped at 1.4 x bandwidth_per_server.')
    parser.add_argument('--comm_contention_fraction', type=float, default=None,
                        help='Fraction r in (0,1] of iter_networking_time '
                             'subject to bandwidth contention. r<1 dampens the '
                             'effective contention factor: effective = 1 + r*(raw-1). '
                             'Default None = legacy (full).')
    parser.add_argument('--cap_n_exponent', type=float, default=None,
                        help='Exponent for N-scaling of the runtime cap: '
                             'cap = alpha * (N/2)^exponent * bandwidth. Default '
                             'None = uniform cap across N.')
    parser.add_argument('--min_guaranteed_bw', type=float, default=None,
                        help='Lower floor (MB/s) for each job using_bandwidth '
                             'when computing contention factor. Default 10.0. '
                             'Lower value allows larger contention factor in '
                             'extreme oversubscription.')
    parser.add_argument('--contention_model', choices=['linear', 'mm1'],
                        default=None,
                        help='Contention factor formula. "linear" (default) '
                             'uses R/U max-min fair share. "mm1" uses M/M/1 PS '
                             'queueing model 1/(1-rho) with rho = total_R/L.')
    parser.add_argument('--mm1_epsilon', type=float, default=None,
                        help='Clamping epsilon for M/M/1 model (default 0.05, '
                             'i.e., max factor = 20).')
    parser.add_argument('--phase_overlap_weighted', action='store_true',
                        help='In iter-model variants, weight non-comm-phase '
                             'jobs by P(in comm) = T_comm/T_iter when computing '
                             'per-server total demand, instead of binary 0/1. '
                             'Captures staggered iteration boundaries.')
    parser.add_argument('--hol_efficiency', type=float, default=None,
                        help='Head-of-line blocking efficiency η ∈ (0,1]. '
                             'When < 1 and contention is active, the contention '
                             'factor is scaled by 1/η. Karol 1987 gives η ≈ 0.586 '
                             'for input-queued switches under random arrivals.')
    return parser


def validate(args: argparse.Namespace) -> None:
    if args.interference_model == 'fixed':
        if args.interference_ratio is None:
            raise SystemExit(
                "Error: --interference-ratio is required when using --interference-model fixed")
    else:
        if args.interference_ratio is not None:
            raise SystemExit(
                f"Error: --interference-ratio must not be used with "
                f"--interference-model {args.interference_model}")

    if args.lcm_time_decimals < 0:
        raise SystemExit("Error: --lcm-time-decimals must be non-negative")

    if not 0.0 <= args.overlapping_ratio <= 1.0:
        raise SystemExit("Error: --overlapping_ratio must be between 0.0 and 1.0 inclusive")

    for label, path in (
        ("--iteration_time_csv_file", args.iteration_time_csv_file),
        ("--communication_volume_csv_file", args.communication_volume_csv_file),
        ("--jobtrace", args.jobtrace),
    ):
        if not Path(path).exists():
            raise SystemExit(f"Error: file for {label} not found: {path}")


def warn_unused_flags(args: argparse.Namespace) -> None:
    """Print a loud warning for flags the chosen interference model never reads.

    These flags are accepted by argparse and forwarded to the Rust core, which
    simply does not consult them for this model. Without this warning the run
    succeeds and the numbers look plausible, so a sensitivity study can measure
    nothing at all and never notice.
    """
    model = args.interference_model
    unused = []

    for flag, dest, default, readers, group in FLAG_READERS:
        if getattr(args, dest) != default and model not in readers:
            unused.append((flag, f"read only by {group}"))

    # `--lcm-algo` is accepted for backward compatibility but the Rust core
    # never reads it at all (see the `lcm_algo` field in mosim/src/config.rs),
    # so every value behaves like the default.
    if args.lcm_algo != 'gcd_fold':
        unused.append(("--lcm-algo", "not implemented in the simulator"))

    if not unused:
        return

    width = max(len(f) for f, _ in unused)
    bar = "=" * 74
    one = len(unused) == 1
    count = "1 flag" if one else f"{len(unused)} flags"
    lines = [
        bar,
        f"WARNING  --interference-model {model} ignores {count} you passed:",
        "",
    ]
    lines += [f"    {flag:<{width}}   {why}" for flag, why in unused]
    lines += [
        "",
        f"Results will be identical to a run without {'it' if one else 'them'}.",
        bar,
    ]
    print("\n".join(lines), file=sys.stderr)


def find_binary() -> str:
    env = os.environ.get('MOSIM_BIN')
    if env:
        return env
    here = Path(__file__).resolve().parent
    candidate = here / 'mosim' / 'target' / 'release' / 'mosim'
    return str(candidate)


def main() -> int:
    args = build_parser().parse_args()
    validate(args)
    warn_unused_flags(args)

    binary = find_binary()
    if not Path(binary).exists():
        raise SystemExit(
            f"Error: mosim binary not found at {binary}\n"
            f"Build it with: (cd mosim && cargo build --release)\n"
            f"Or set MOSIM_BIN to its path.")

    config_json = json.dumps(vars(args))
    t0 = time.perf_counter()
    proc = subprocess.run([binary], input=config_json, text=True, check=False)
    cli_runtime_seconds = time.perf_counter() - t0

    # Append the wrapper-side wall-clock runtime to the result log. This
    # includes Rust binary startup + stdin parsing + file I/O on top of
    # the inner `sim_runtime_seconds` that the Rust core writes.
    if proc.returncode == 0:
        log_path = Path(args.log)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"cli_runtime_seconds: {cli_runtime_seconds:.6f}\n")
        except OSError as e:
            print(
                f"simulator-trace-timer-bw: failed to append cli_runtime_seconds "
                f"to {log_path}: {e}",
                file=sys.stderr,
            )
    print(f"cli_runtime_seconds: {cli_runtime_seconds:.6f}", file=sys.stderr)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
