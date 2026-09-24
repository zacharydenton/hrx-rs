"""Qualification of paired measurements; collection stays with each workload.

Missing, failed, contended, non-finite, or numerically unqualified runs cannot
produce a promotion. This module performs no device work and writes no files.
"""
import math
import statistics


def assess_pairs(runs, pairs, *, primary=False, runtime=False, memory=False):
    """Assess records with pair/arm, milliseconds, quality_passed and uncontended.

    Memory candidates additionally supply peak_bytes. Optional stricter
    workload limits should be applied by the caller, never replaced here.
    """
    minimum = 5 if runtime else 3
    reasons = []
    if pairs < minimum:
        reasons.append(f"requires at least {minimum} process pairs")
    slots = {}
    for run in runs:
        key = (run.get("pair"), run.get("arm"))
        if key in slots:
            reasons.append("duplicate arm in a pair")
        slots[key] = run
    expected = {(pair, arm) for pair in range(pairs)
                for arm in ("baseline", "candidate")}
    if set(slots) != expected:
        reasons.append("incomplete or unexpected pairs")
    for run in runs:
        if run.get("exit_code") != 0:
            reasons.append("failed execution")
        if run.get("quality_passed") is not True:
            reasons.append("missing or failed numerical qualification")
        if run.get("uncontended") is not True:
            reasons.append("exclusive device use not established")
        value = run.get("milliseconds")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            reasons.append("invalid timing")
        if memory and (not isinstance(run.get("peak_bytes"), int) or isinstance(run["peak_bytes"], bool) or run["peak_bytes"] <= 0):
            reasons.append("missing peak allocation measurement")
    if reasons:
        return {"accepted": False, "reasons": sorted(set(reasons))}
    ratios = [slots[pair, "candidate"]["milliseconds"] /
              slots[pair, "baseline"]["milliseconds"] for pair in range(pairs)]
    limit = 1.05 if runtime and not memory else 1.03
    if primary and not memory:
        limit = 0.95
    median = statistics.median(ratios)
    if median > limit:
        reasons.append("latency target not met")
    if primary and not memory and sum(r < 1 for r in ratios) < math.ceil(2 * pairs / 3):
        reasons.append("improvement not consistent across pairs")
    result = {"paired_time_ratios": ratios, "median_time_ratio": median,
              "time_ratio_limit": limit}
    if memory:
        peaks = [slots[pair, "candidate"]["peak_bytes"] /
                 slots[pair, "baseline"]["peak_bytes"] for pair in range(pairs)]
        result["paired_peak_ratios"] = peaks
        if statistics.median(peaks) >= 1:
            reasons.append("peak allocation did not decrease")
    result.update(accepted=not reasons, reasons=reasons)
    return result


def validate_identity(manifest, runs):
    """Validate a saved-build manifest against the binaries actually measured.

    Require complete evidence instead of inferring a saved binary's identity
    from whatever checkout happens to exist at measurement time.
    """
    required = ("source_revision", "source_tree_sha256", "binary_sha256", "native_hashes", "compiler_sha256", "rustc_identity")
    for arm in ("baseline", "candidate"):
        identity = manifest.get(arm, {})
        if any(not identity.get(key) for key in required):
            raise ValueError(f"incomplete {arm} build identity")
        for run in runs:
            if run["arm"] == arm and run.get("binary_sha256") != identity["binary_sha256"]:
                raise ValueError(f"{arm} binary does not match its build identity")
            if run["arm"] == arm and run.get("runtime_sha256") != identity["native_hashes"]:
                raise ValueError(f"{arm} native libraries do not match their build identity")
    for field in ("workload", "model_identity", "input_identity", "cache_state", "allocation_policy", "timed_scope", "hardware_identity", "oracle_identity"):
        if not manifest.get(field):
            raise ValueError(f"missing comparison identity: {field}")
    for arm in ("baseline", "candidate"):
        if not isinstance(manifest[arm].get("build_environment"), dict):
            raise ValueError(f"missing {arm} build environment")
    for field in ("rustc_identity", "build_environment"):
        if manifest["baseline"][field] != manifest["candidate"][field]:
            raise ValueError(f"comparison changed {field}")
    kind = manifest.get("comparison_kind")
    if kind == "application":
        for field in ("native_hashes", "compiler_sha256"):
            if manifest["baseline"][field] != manifest["candidate"][field]:
                raise ValueError(f"application comparison changed {field}")
    elif kind == "runtime":
        for field in ("source_revision", "source_tree_sha256"):
            if manifest["baseline"][field] != manifest["candidate"][field]:
                raise ValueError(f"runtime comparison changed the application {field}")
    else:
        raise ValueError("comparison_kind must be application or runtime")
