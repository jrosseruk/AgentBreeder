from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import Score, scorer, accuracy

from typing import Any, Literal, Union
from textwrap import dedent

from evals.benchmark import Benchmark, register_benchmark
from evals.metrics import ci_lower, ci_upper, median

from sympy import Eq, sympify, simplify
from sympy.parsing.latex import parse_latex


@register_benchmark("math")
class Math(Benchmark):

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:
        self.split = split

        self.args = args

        split_mapping = {
            "validation": "train",
            "test": "test",
        }

        self.dataset = self.filtered_hf_dataset(
            path="Maxwell-Jia/MATH",
            name="default",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def benchmark_filter(self, example):
        if example["level"] == "Level 5":
            return True

        return False

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following maths question:

            {record["problem"]}
        """
        ).strip()

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n\n"
        output_format = "Provide your final answer as only a latex string, this could be, e.g. a single number, equation, or a short piece of text. Don't include units e.g. cm, degrees etc."
        prompt += f"OUTPUT ANSWER FORMAT: {output_format}"

        boxed = Math.extract_boxed_content(record["solution"])[0]
        # print(boxed, record["solution"].split("\\boxed")[1])
        # print(boxed)

        return Sample(
            input=prompt,
            target=boxed,
            metadata={
                "format": output_format,
                "solution": record["solution"],
                "level": record["level"],
                "unique_id": record["unique_id"],
            },
        )

    @staticmethod
    def extract_boxed_content(input_str):
        """
        Extracts all contents inside \boxed{...} from the input string.

        Args:
            input_str (str): The input string containing LaTeX code.

        Returns:
            list: A list of strings extracted from within each \boxed{...}.
        """
        boxed_contents = []
        search_str = "\\boxed{"
        start = 0

        while True:
            # Find the next occurrence of \boxed{
            idx = input_str.find(search_str, start)
            if idx == -1:
                break  # No more \boxed{ found

            # Initialize stack to handle nested braces
            stack = []
            content_start = idx + len(search_str)
            i = content_start

            while i < len(input_str):
                char = input_str[i]
                if char == "{":
                    stack.append("{")
                elif char == "}":
                    if stack:
                        stack.pop()
                    else:
                        # Matching closing brace for \boxed{
                        content_end = i
                        boxed_content = input_str[content_start:content_end]
                        boxed_contents.append(boxed_content)
                        start = i + 1  # Update start position for next search
                        break
                i += 1
            else:
                # If loop completes without finding a matching '}', raise an error
                raise ValueError("Unmatched '{' found in the input string.")

        return boxed_contents

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def math_match():
        async def score(state, target):
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="math_match",
                    value=0,
                    answer=state.output.completion,
                    explanation=f"Error in model response.",
                )

            try:
                accuracy = Math.match_latex(target.text, state.output.completion)
                if accuracy:
                    accuracy = 1
                else:
                    # print(
                    #     "target.text",
                    #     target.text,
                    #     target.text[6:-1],
                    #     "state.output.completion",
                    #     state.output.completion,
                    # )

                    if target.text[6:-1] == state.output.completion:
                        accuracy = 1
                    else:
                        if "=" in state.output.completion:
                            rhs = state.output.completion.split("=")[-1]
                            print(target.text, rhs)
                            accuracy = Math.match_latex(target.text, rhs)
                            if accuracy:
                                accuracy = 1
                            else:
                                accuracy = 0
                        else:
                            accuracy = 0

            except Exception as e:
                accuracy = 0

            return Score(
                name="math_match",
                value=accuracy,
                answer=state.output.completion,
                explanation=f"Math match",
            )

        return score

    @task
    def match_task(self):
        return Task(
            time_limit=self.args.task_timeout,
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.math_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    def parse_single_expr(s: str):
        """
        Parse a single LaTeX or plain mathematical expression into a simplified SymPy object.

        Args:
            s (str): The expression to parse.

        Returns:
            SymPy expression.
        """
        if not s:
            return None  # Handle empty strings gracefully

        try:
            expr = parse_latex(s)
        except Exception:
            try:
                expr = sympify(s)
            except Exception as e:
                raise ValueError(f"Failed to parse expression '{s}': {e}")

        return simplify(expr)

    @staticmethod
    def strip_dollar(s: str) -> str:
        """
        Strip surrounding dollar signs from a LaTeX expression if present.

        Args:
            s (str): The expression string.

        Returns:
            str: The expression without surrounding dollar signs.
        """
        s = s.strip()
        if s.startswith("$") and s.endswith("$"):
            return s[1:-1]
        return s

    @staticmethod
    def parse_expression(s: str):
        """
        Parse an expression which can be a single expression or a comma-separated list.

        Args:
            s (str): The expression string.

        Returns:
            SymPy expression or tuple of SymPy expressions.
        """
        s = s.strip()
        if not s:
            return tuple()  # Return an empty tuple for empty strings

        if "," in s:
            # Split by commas and parse each element
            elements = [Math.parse_single_expr(e.strip()) for e in s.split(",")]
            return tuple(e for e in elements if e is not None)
        else:
            return Math.parse_single_expr(s)

    @staticmethod
    def compare_expression_lists(list1: tuple, list2: tuple) -> bool:
        """
        Compare two lists of SymPy expressions for equivalence, ignoring order.

        Args:
            list1 (tuple): First list of SymPy expressions.
            list2 (tuple): Second list of SymPy expressions.

        Returns:
            bool: True if lists are equivalent as multisets, False otherwise.
        """
        # Create a mutable copy of list2's expressions to track matches
        unmatched = list(list2)

        for expr1 in list1:
            match_found = False
            for idx, expr2 in enumerate(unmatched):
                if expr1.equals(expr2):
                    # Match found; remove from unmatched and break
                    del unmatched[idx]
                    match_found = True
                    break
            if not match_found:
                # No matching expression found for expr1
                return False

        # All expressions in list1 have been matched; ensure list2 has no extra elements
        return len(unmatched) == 0

    @staticmethod
    def match_latex(expr1: str, expr2: str) -> bool:
        """
        Compare two LaTeX expressions or lists for mathematical equivalence.

        Args:
            expr1 (str): First LaTeX expression or list.
            expr2 (str): Second LaTeX expression or list.

        Returns:
            bool: True if the expressions or lists are equivalent, False otherwise.
        """
        try:
            # Strip dollar signs if present
            expr1 = Math.strip_dollar(expr1)
            expr2 = Math.strip_dollar(expr2)

            # Parse both expressions
            sympy_expr1 = Math.parse_expression(expr1)
            sympy_expr2 = Math.parse_expression(expr2)

            # Compare the two parsed expressions
            if isinstance(sympy_expr1, tuple) and isinstance(sympy_expr2, tuple):
                if len(sympy_expr1) != len(sympy_expr2):
                    return False
                return Math.compare_expression_lists(sympy_expr1, sympy_expr2)
            elif isinstance(sympy_expr1, tuple) or isinstance(sympy_expr2, tuple):
                # One is a list and the other is a single expression
                return False
            else:
                # Both are single expressions
                return sympy_expr1.equals(sympy_expr2)

        except Exception as e:
            raise ValueError(f"Error parsing LaTeX expressions: {e}")


if __name__ == "__main__":
    # Define test cases as tuples of (expr1, expr2)
    test_cases = [
        (r"\frac{1}{2} + \frac{1}{3}", r"1"),  # Test Case 1: Not Equivalent
        (r"$\frac{1}{2} + \frac{1}{2}$", r"1"),  # Test Case 2: Equivalent
        ("2, 3, 4", "2, 5"),  # Test Case 3: Not Equivalent
        ("2, 3, 4", "3, 2, 4"),  # Test Case 4: Equivalent (Permutation)
        (r"\sin^2{x} + \cos^2{x}", "1"),  # Test Case 5: Equivalent
        (r"$x^2 - y^2$", r"(x - y)(x + y)"),  # Test Case 6: Equivalent
        ("a, b, c", "a, b, c"),  # Test Case 7: Equivalent
        ("a, b, c", "a, b"),  # Test Case 8: Not Equivalent
        (r"\sqrt{4}", "2"),  # Test Case 9: Equivalent
        (r"$\sqrt{4}$", "2"),  # Test Case 10: Equivalent
        ("2, 2, 3", "2, 3, 2"),  # Test Case 11: Equivalent with duplicates
        ("2, 2, 3", "2, 3, 3"),  # Test Case 12: Not Equivalent with duplicates
        (r"e^{i\pi} + 1", "0"),  # Test Case 13: Equivalent (Euler's identity)
        ("x + y, y + x", "2*x + 2*y"),  # Test Case 14: Not Equivalent
        ("x + y, y + x", "x + y, x + y"),  # Test Case 15: Equivalent with duplicates
        ("", ""),  # Test Case 16: Both empty (Equivalent)
        ("", "0"),  # Test Case 17: One empty, one not (Not Equivalent)
    ]

    matcher = Math()

    for idx, (expr1, expr2) in enumerate(test_cases, start=1):
        try:
            is_equal = matcher.match_latex(expr1, expr2)
            result = "Equivalent" if is_equal else "Not Equivalent"
        except ValueError as ve:
            result = f"Error: {ve}"
        print(f"Test Case {idx}: {result}", f"([{expr1}], [{expr2}])")

    @classmethod
    @property
    def prompt(self):
        return dedent(
            """
        Your aim is to find an optimal multi-agent scaffold performing well on the Math benchmark, which evaluates
        mathematical problem-solving abilities across a wide range of topics and difficulty levels.
        This benchmark includes problems from algebra, geometry, calculus, number theory, and more,
        designed to test both computational skills and conceptual understanding.
        ## An example question from Math:
        **Question**: Solve the following mathematical problem.
        A car travels 100 miles at a speed of 50 mph. How many hours does it take to travel 100 miles?
        **Answer (Not Given)**: 2
        """
        )
