"""Sanity checks for held-out evaluation; do not assert a fixed accuracy."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("validation_script", Path(__file__).with_name("validate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationTests(unittest.TestCase):
    def test_confusion_matrix_and_fpr(self):
        m = module.scores([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual([m[k] for k in ("tp", "fp", "tn", "fn")], [1, 1, 1, 1])
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["false_positive_rate"], 0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)

    def test_all_normal_denominators(self):
        m = module.scores([0, 0], [0, 0])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["false_positive_rate"], 0.0)
        self.assertIsNone(m["recall"])

    def test_reproducible_split_and_confusion_counts(self):
        a = module.evaluate(samples=1000, seed=12)
        b = module.evaluate(samples=1000, seed=12)
        self.assertEqual(a, b)
        overall = a["overall"]
        self.assertEqual(overall["samples"], sum(overall[k] for k in ("tp", "fp", "tn", "fn")))
        self.assertTrue(0 <= overall["false_positive_rate"] <= 1)
        self.assertGreater(overall["samples"], 0)


if __name__ == "__main__":
    unittest.main()
