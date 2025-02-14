import sys

sys.path.append("src")

from base import System
import unittest
from benchmarks.mmlu_cf import MMLUCF
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import random


class TestMMLUCF(unittest.TestCase):

    def setUp(self):
        self.system = System(
            system_name="clrs_text_test_system",
            system_id="test_id",
            system_code=dedent(
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
        self.evaluator = MMLUCF(
            args=self.args, split="validation", shuffle=False, limit=100
        )
        print(self.evaluator.dataset[0].metadata["unique_id"])
        print(random.choice(self.evaluator.dataset))
        # print([sample.input for sample in self.evaluator.dataset])


if __name__ == "__main__":
    unittest.main()
