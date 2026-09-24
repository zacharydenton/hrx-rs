import copy
import unittest
from performance_evidence import assess_pairs, validate_identity


def runs(pairs=3):
    return [dict(pair=i, arm=arm, milliseconds=100 if arm == "baseline" else 90,
                 exit_code=0, quality_passed=True, uncontended=True,
                 binary_sha256=arm, runtime_sha256={"lib": "hash"})
            for i in range(pairs) for arm in ("baseline", "candidate")]


class EvidenceTests(unittest.TestCase):
    def test_primary_control_and_runtime_thresholds(self):
        self.assertTrue(assess_pairs(runs(), 3, primary=True)["accepted"])
        data = runs()
        for run in data:
            if run["arm"] == "candidate":
                run["milliseconds"] = 104
        self.assertFalse(assess_pairs(data, 3)["accepted"])
        self.assertFalse(assess_pairs(runs(), 3, runtime=True)["accepted"])
        self.assertTrue(assess_pairs(runs(5), 5, runtime=True)["accepted"])

    def test_incomplete_duplicate_failure_noise_and_missing_quality(self):
        for field, value in (("milliseconds", float("nan")), ("milliseconds", 0),
                             ("quality_passed", None), ("uncontended", False),
                             ("exit_code", 1)):
            data = runs()
            data[0][field] = value
            self.assertFalse(assess_pairs(data, 3)["accepted"], field)
        self.assertFalse(assess_pairs(runs()[:-1], 3)["accepted"])
        self.assertFalse(assess_pairs(runs() + [runs()[0]], 3)["accepted"])

    def test_memory_requires_reduction_and_latency_budget(self):
        data = runs()
        for run in data:
            run["peak_bytes"] = 200 if run["arm"] == "baseline" else 100
        self.assertTrue(assess_pairs(data, 3, primary=True, memory=True)["accepted"])
        data[1]["peak_bytes"] = data[3]["peak_bytes"] = 300
        self.assertFalse(assess_pairs(data, 3, memory=True)["accepted"])

    def test_identity_prevents_misattributed_comparisons(self):
        manifest = {key: "fixed" for key in ("workload", "model_identity", "input_identity", "cache_state", "allocation_policy", "timed_scope", "hardware_identity", "oracle_identity")}
        manifest.update(comparison_kind="application")
        for arm in ("baseline", "candidate"):
            manifest[arm] = dict(source_revision=arm, source_tree_sha256=arm, binary_sha256=arm,
                                 native_hashes={"lib": "hash"}, compiler_sha256="compiler",
                                 rustc_identity="rustc", build_environment={})
        validate_identity(manifest, runs())
        for field in ("native_hashes", "compiler_sha256", "binary_sha256", "rustc_identity", "build_environment"):
            changed = copy.deepcopy(manifest)
            changed["candidate"][field] = "different"
            with self.assertRaises(ValueError):
                validate_identity(changed, runs())
        manifest['comparison_kind'] = 'runtime'
        manifest['candidate']['source_revision'] = manifest['baseline']['source_revision']
        with self.assertRaises(ValueError):
            validate_identity(manifest, runs())
        manifest['candidate']['source_tree_sha256'] = manifest['baseline']['source_tree_sha256']
        validate_identity(manifest, runs())


if __name__ == "__main__":
    unittest.main()
