import sys

sys.path.append("src")

from base import Scaffold
import unittest
from evals.benchmarks.drop import DROP, drop_metric
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from discover.seed_scaffolds import COT_SC
import re
import asyncio


class TestDROP(unittest.TestCase):

    def setUp(self):
        self.scaffold = Scaffold(
            scaffold_name="clrs_text_test_scaffold",
            scaffold_id="test_id",
            scaffold_code=dedent(
                """
            async def forward(self, task, required_answer_format):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--random_seed", type=int, default=42)

        self.args = parser.parse_args()

    def test_record_to_sample(self):
        self.evaluator = DROP(args=self.args, split="validation", limit=1)

    def test_exact_match_single_reference(self):
        sample = "Dockers, Eagles"
        gold_reference = ["Dockers", "Eagles"]
        em, f1 = drop_metric(sample, gold_reference)
        self.assertEqual(em, 1.0)
        self.assertEqual(f1, 100.0)

    def test2(self):
        sample = "duke of york", "king of england"


if __name__ == "__main__":
    unittest.main()
