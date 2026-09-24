"""Workload aliases and timing adapters for saved consumer campaigns."""
import math
import re
import time


def wait_for_idle(read_busy, *, timeout=30., sleep=time.sleep, clock=time.monotonic):
    """Require three idle samples so the preceding run's busy counter can decay."""
    started = clock()
    consecutive = 0
    while True:
        busy = read_busy().strip()
        consecutive = consecutive + 1 if busy == "0" else 0
        elapsed = clock() - started
        if consecutive == 3 or elapsed >= timeout:
            return {"idle": consecutive == 3, "last_busy": busy,
                    "wait_seconds": elapsed, "idle_samples": consecutive}
        sleep(0.1)


def configure_workloads(plans, configuration=None):
    """Select built-in consumers with optional explicit argument vectors.

    Configuration maps workload IDs to consumer names and optional arguments.
    The literal {output_dir} in arguments expands to that invocation's directory.
    """
    if configuration is None:
        return plans, {name: name for name in plans}
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError("workloads must be a nonempty object")
    selected, consumers = {}, {}
    for name, entry in configuration.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("workload IDs must contain only letters, digits, hyphens and underscores")
        if not isinstance(entry, dict) or set(entry) - {"consumer", "arguments"}:
            raise ValueError(f"invalid workload description: {name}")
        consumer = entry.get("consumer")
        if not isinstance(consumer, str) or consumer not in plans:
            raise ValueError(f"unknown consumer in workload {name}")
        repo, binary, flags = plans[consumer]
        if "arguments" in entry:
            arguments = entry["arguments"]
            if not isinstance(arguments, list) or not all(isinstance(a, str) and "\0" not in a for a in arguments):
                raise ValueError(f"arguments must be a string array: {name}")
            arguments = tuple(arguments)
            flags = lambda out, arguments=arguments: [a.replace("{output_dir}", str(out)) for a in arguments]
        selected[name] = (repo, binary, flags)
        consumers[name] = consumer
    return selected, consumers


def timing_ms(consumer, reports, file_report=None):
    """Read an explicitly completed timing from the consumer's report schema."""
    if consumer == "hrxdb":
        if not isinstance(file_report, dict):
            raise ValueError("HRXDB report must be an object")
        batch = file_report.get("batch", 1)
        if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 64:
            raise ValueError("invalid HRXDB report batch size")
        if batch > 1:
            value = file_report["batch_median_ms"]
        else:
            trials = file_report["trials"]
            if len(trials) != 1:
                raise ValueError("qualification requires one HRXDB schedule per workload")
            value = trials[0]["search_median_ms"]
    else:
        value = reports[-1]["median_ms"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("missing, non-finite or non-positive timing")
    return value
