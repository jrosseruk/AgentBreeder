import sys

sys.path.append("src")

from base import Scaffold
import unittest
from evals.benchmarks.mmlu import MMLU
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm


class TestMMLU(unittest.TestCase):

    def setUp(self):
        self.scaffold = Scaffold(
            scaffold_name="test_scaffold",
            scaffold_id="test_id",
            scaffold_code=dedent(
                """
            def forward(self, task):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--db_name", type=str, default="illuminator.db")

        self.args = parser.parse_args()

        self.evaluator = MMLU(self.args)

    def test_record_to_sample(self):
        record = {
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": 2,
            "subject": "Geography",
        }
        sample = self.evaluator._record_to_sample(record)
        expected_prompt = dedent(
            """
            Answer the following multiple choice question.

            What is the capital of France?
            (A) Berlin
            (B) Madrid
            (C) Paris
            (D) Rome

            Provide your answer as a single letter in the range A-D.
        """
        ).strip()
        self.assertEqual(sample.input, expected_prompt)
        self.assertEqual(sample.target, "C")
        self.assertEqual(sample.metadata["subject"], "Geography")

    # def test_evaluate(self):
    #     accuracy = self.evaluator.evaluate(self.scaffold, limit=1000)
    #     self.assertIsInstance(accuracy, float)

    def test_evaluate_multiple(self):

        scaffold_5 = Scaffold(
            scaffold_name="test_scaffold",
            scaffold_id="test_id",
            scaffold_code="""async def forward(self, task: str) -> str:

    # import time
    # time.sleep(5)
    
    return "C"
""",
        )

        scaffold_10 = Scaffold(
            scaffold_name="test_scaffold",
            scaffold_id="test_id",
            scaffold_code="""async def forward(self, task: str) -> str:

    # import time
    # time.sleep(2)

    import asyncio
    await asyncio.sleep(2)
    
    
    return 'C'
""",
        )
        scaffolds = [scaffold_10]

        for scaffold in tqdm(scaffolds, total=len(scaffolds)):
            evaluator = MMLU(self.args)
            accuracy = evaluator.evaluate(scaffold, limit=5)
            self.assertIsInstance(accuracy, float)


if __name__ == "__main__":
    unittest.main()
