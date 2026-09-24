import unittest
from pathlib import Path
from consumer_workloads import configure_workloads, timing_ms, wait_for_idle


class WorkloadTests(unittest.TestCase):
    def test_qwen_uses_completed_generation_seconds_from_sidecar(self):
        self.assertEqual(timing_ms("qwen_generation", [], {"elapsed_seconds": 1.25}), 1250.)
        for value in [True, 0, -1, float("nan"), float("inf"), 1e308, "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_ms("qwen_generation", [], {"elapsed_seconds": value})
        with self.assertRaises(ValueError):
            timing_ms("qwen_generation", [], None)

    def test_idle_requires_consecutive_samples_and_times_out(self):
        now = [0.]
        def sleep(seconds):
            now[0] += seconds
        values = iter(["7", "0", "0", "1", "0", "0", "0"])
        result = wait_for_idle(lambda: next(values), sleep=sleep, clock=lambda: now[0])
        self.assertTrue(result["idle"])
        self.assertEqual(result["idle_samples"], 3)
        result = wait_for_idle(lambda: "9", timeout=.3, sleep=sleep, clock=lambda: now[0])
        self.assertFalse(result["idle"])
        self.assertGreaterEqual(result["wait_seconds"], .3)

    def test_aliases_keep_consumer_identity_and_separate_output_paths(self):
        plans = {"hrxdb": ("hrxdb", "hrxdb-bench", lambda out: ["--output", str(out / "report.json")])}
        selected, consumers = configure_workloads(plans, {
            "batch3": {"consumer": "hrxdb", "arguments": ["--batch", "3", "--output", "{output_dir}/report.json"]},
            "control": {"consumer": "hrxdb"},
        })
        self.assertEqual(consumers, {"batch3": "hrxdb", "control": "hrxdb"})
        self.assertEqual(selected["batch3"][2](Path("a")), ["--batch", "3", "--output", "a/report.json"])
        self.assertEqual(selected["control"][2](Path("b")), ["--output", "b/report.json"])
        self.assertIs(configure_workloads(plans)[0], plans)

    def test_invalid_workloads_fail_before_execution(self):
        plans = {"hrxdb": ("hrxdb", "bench", lambda out: [])}
        for configuration in [[], {}, {"../escape": {"consumer": "hrxdb"}},
                              {"x": {"consumer": "unknown"}}, {"x": {"consumer": []}},
                              {"x": {"consumer": "hrxdb", "arguments": "--batch 3"}},
                              {"x": {"consumer": "hrxdb", "arguments": [3]}},
                              {"x": {"consumer": "hrxdb", "argument": []}}]:
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                configure_workloads(plans, configuration)

    def test_batch_adapter_uses_batch_latency_not_sequential_control(self):
        self.assertEqual(timing_ms("hrxdb", [], {"batch": 3, "batch_median_ms": 2., "sequential_median_ms": 9.}), 2.)
        self.assertEqual(timing_ms("hrxdb", [], {"trials": [{"search_median_ms": 4.}]}), 4.)
        self.assertEqual(timing_ms("dinov3", [{"median_ms": 3.}]), 3.)
        with self.assertRaises(ValueError):
            timing_ms("hrxdb", [], {"trials": [{"search_median_ms": 4.}] * 2})
        for value in [True, 0, -1, float("nan"), float("inf"), "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_ms("hrxdb", [], {"batch": 3, "batch_median_ms": value})
        for report in [None, [], {"batch": True}, {"batch": "3"}, {"batch": 0}]:
            with self.subTest(report=report), self.assertRaises(ValueError):
                timing_ms("hrxdb", [], report)


if __name__ == "__main__":
    unittest.main()
