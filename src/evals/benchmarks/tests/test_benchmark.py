import sys
import unittest

sys.path.append("src")
import argparse

from evals.benchmarks.math_500 import Math500
from evals.negative_sampler import get_positive_and_negative_samples


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--random_seed", type=int, default=42)
        args = parser.parse_args()
        cls.math_500 = Math500(args, split="validation", shuffle=True, limit=1000)
        cls.positive_negative_samples = get_positive_and_negative_samples("Math500")

    def test_positive_samples(self):
        positive_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] in self.positive_negative_samples[1.0]
        )
        self.assertGreater(positive_count, 0, "No positive samples found")

    def test_negative_samples(self):
        negative_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] in self.positive_negative_samples[0]
        )
        self.assertGreater(negative_count, 0, "No negative samples found")

    def test_unlabeled_samples(self):
        unlabeled_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] not in self.positive_negative_samples[1.0]
            and record.metadata["unique_id"] not in self.positive_negative_samples[0]
        )
        self.assertGreater(unlabeled_count, 0, "No unlabeled samples found")

    def test_total_samples(self):
        total_count = len(self.math_500.dataset)
        positive_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] in self.positive_negative_samples[1.0]
        )
        negative_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] in self.positive_negative_samples[0]
        )
        unlabeled_count = sum(
            1
            for record in self.math_500.dataset
            if record.metadata["unique_id"] not in self.positive_negative_samples[1.0]
            and record.metadata["unique_id"] not in self.positive_negative_samples[0]
        )
        self.assertEqual(
            total_count,
            positive_count + negative_count + unlabeled_count,
            "Total count does not match sum of positive, negative, and unlabeled counts",
        )


if __name__ == "__main__":
    unittest.main()
