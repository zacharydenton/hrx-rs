#!/usr/bin/env python3
"""Measure whether GPU and NPU work contends on Strix Halo's shared memory controller.

The iGPU (gfx1151) and XDNA2 NPU sit behind one LPDDR5X controller, so running both at
once does not necessarily add throughput. This runs each load alone to establish a
baseline, then both concurrently, and reports the *concurrency efficiency*:

    efficiency = gpu_concurrent/gpu_alone + npu_concurrent/npu_alone

1.0 means the devices are purely zero-sum -- whatever the NPU gains, the GPU loses, and
heterogeneous execution buys nothing. 2.0 would mean perfect independence. The value in
between is the real headroom for splitting a model across both engines.

Each load generator prints one JSON object on stdout.
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GPU = HERE / "target/release/examples/gpu_load"
DEFAULT_NPU = pathlib.Path.home() / "code/dinov3-xdna2/target/release/npu_load"
DEFAULT_NPU_BUILD = pathlib.Path.home() / "code/dinov3-xdna2/build"


def gpu_command(options):
    return [
        str(options.gpu),
        "--seconds", str(options.seconds),
        "--mib", str(options.mib),
        "--mode", options.mode,
    ]


def npu_command(options):
    return [
        str(options.npu), str(options.npu_build),
        "--seconds", str(options.seconds),
        "--m", str(options.m),
    ]


def launch(command):
    return subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def collect(process, label):
    """Wait for a load generator and return its JSON record."""
    out, err = process.communicate()
    if process.returncode != 0:
        sys.exit(f"{label} failed (exit {process.returncode}):\n{err.strip()}")
    line = next((l for l in out.splitlines() if l.startswith("{")), None)
    if line is None:
        sys.exit(f"{label} printed no JSON record:\n{out.strip()}\n{err.strip()}")
    return json.loads(line)


def run_alone(command, label):
    return collect(launch(command), label)


def run_together(gpu, npu):
    """Start both loads as close to simultaneously as possible and let them overlap.

    Both run for the same wall-clock window, so each spends nearly all of its measured
    time alongside the other; the brief unpaired head and tail bias the result *toward*
    optimism, which is the safe direction for a go/no-go probe.
    """
    gpu_process = launch(gpu)
    npu_process = launch(npu)
    return collect(gpu_process, "gpu_load"), collect(npu_process, "npu_load")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=3, help="trials per configuration")
    parser.add_argument("--mib", type=int, default=256, help="GPU buffer size")
    parser.add_argument("--mode", default="fill", choices=("fill", "copy"))
    parser.add_argument("--m", type=int, default=16384, help="NPU GEMM rows")
    parser.add_argument("--gpu", type=pathlib.Path, default=DEFAULT_GPU)
    parser.add_argument("--npu", type=pathlib.Path, default=DEFAULT_NPU)
    parser.add_argument("--npu-build", type=pathlib.Path, default=DEFAULT_NPU_BUILD)
    parser.add_argument("--json", type=pathlib.Path, help="write the raw records here")
    options = parser.parse_args()

    for binary in (options.gpu, options.npu):
        if not binary.exists():
            sys.exit(f"missing load generator: {binary}")

    gpu, npu = gpu_command(options), npu_command(options)
    records = {"alone": {"gpu": [], "npu": []}, "together": {"gpu": [], "npu": []}}

    for trial in range(options.repeat):
        print(f"trial {trial + 1}/{options.repeat}", file=sys.stderr)
        # Alone first, and each device separately, so neither baseline sees the other.
        records["alone"]["gpu"].append(run_alone(gpu, "gpu_load"))
        records["alone"]["npu"].append(run_alone(npu, "npu_load"))
        together_gpu, together_npu = run_together(gpu, npu)
        records["together"]["gpu"].append(together_gpu)
        records["together"]["npu"].append(together_npu)

    def median(kind, device, field):
        return statistics.median(r[field] for r in records[kind][device])

    gpu_alone = median("alone", "gpu", "gb_s_mean")
    gpu_together = median("together", "gpu", "gb_s_mean")
    npu_alone = median("alone", "npu", "dispatch_s")
    npu_together = median("together", "npu", "dispatch_s")
    npu_bw_alone = median("alone", "npu", "gb_s")
    npu_bw_together = median("together", "npu", "gb_s")

    gpu_ratio = gpu_together / gpu_alone
    npu_ratio = npu_together / npu_alone
    efficiency = gpu_ratio + npu_ratio

    print()
    print(f"GPU load: {options.mode} over {options.mib} MiB")
    print(f"NPU load: bfp16 GEMM m={options.m} k=4096 n=1024")
    print(f"{options.repeat} trials of {options.seconds:g}s each\n")
    row = "{:<26} {:>12} {:>12} {:>9}"
    print(row.format("metric", "alone", "concurrent", "retained"))
    print("-" * 62)
    print(row.format(
        "GPU bandwidth (GB/s)", f"{gpu_alone:.1f}", f"{gpu_together:.1f}",
        f"{gpu_ratio:.1%}",
    ))
    print(row.format(
        "NPU dispatch (per s)", f"{npu_alone:.1f}", f"{npu_together:.1f}",
        f"{npu_ratio:.1%}",
    ))
    print(row.format(
        "NPU bandwidth (GB/s)", f"{npu_bw_alone:.1f}", f"{npu_bw_together:.1f}", "",
    ))
    print("-" * 62)
    print(f"{'concurrency efficiency':<26} {efficiency:>12.2f}   (1.00 = zero-sum, 2.00 = independent)")
    print(f"{'aggregate GB/s':<26} {gpu_alone + npu_bw_alone:>12.1f} {gpu_together + npu_bw_together:>12.1f}")

    if options.json:
        options.json.write_text(json.dumps(records, indent=2))
        print(f"\nraw records written to {options.json}")


if __name__ == "__main__":
    main()
