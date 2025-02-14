import sys

sys.path.append("src")

from base import Scaffold
import unittest
from benchmarks.salad_data import SaladData
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session


class TestSaladData(unittest.TestCase):

    def setUp(self):
        self.scaffold = Scaffold(
            scaffold_name="salad_data_test_scaffold",
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
        for session in initialize_session():
            scaffolds_for_validation = session.query(Scaffold).limit(1).all()
            print(scaffolds_for_validation[0].scaffold_id)
            self.evaluator = SaladData(
                args=self.args, split="test", shuffle=False, limit=100
            )
            print([sample.input[30] for sample in self.evaluator.dataset])


if __name__ == "__main__":
    unittest.main()
