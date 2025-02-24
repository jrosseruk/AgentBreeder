import sys
import unittest
import asyncio

sys.path.append("src")
from api.completion import get_json_completion


class TestCompletion(unittest.TestCase):
    def setUp(self):
        self.test_messages = [
            {
                "role": "user",
                "content": "Please think step by step and then solve the task.",
            },
            {
                "role": "user",
                "content": "What is the capital of France? A: Paris B: London C: Berlin D: Madrid.",
            },
        ]

        self.test_format = {
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
        }

    def test_claude_completion(self):
        response = asyncio.run(
            get_json_completion(
                messages=self.test_messages,
                response_format=self.test_format,
                model="claude-3-sonnet-20240229",
            )
        )

        self.assertIn("thinking", response)
        self.assertIn("answer", response)
        self.assertIn(response["answer"], ["A", "B", "C", "D"])

    def test_gpt_completion(self):
        response = asyncio.run(
            get_json_completion(
                messages=self.test_messages,
                response_format=self.test_format,
                model="gpt-4",
            )
        )

        self.assertIn("thinking", response)
        self.assertIn("answer", response)
        self.assertIn(response["answer"], ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
