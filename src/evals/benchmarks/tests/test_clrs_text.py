import sys

sys.path.append("src")

from base import Scaffold
import unittest
from evals.benchmarks.clrs_text import CLRSText
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from discover.seed_scaffolds import COT_SC
import re
import asyncio


class TestCLRSText(unittest.TestCase):

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
        self.evaluator = CLRSText(args=self.args, split="validation", limit=1)


if __name__ == "__main__":
    unittest.main()
