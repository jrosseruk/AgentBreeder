from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from benchmarks.benchmark import Benchmark, register_benchmark


@register_benchmark("mmlu_cf")
class MMLUCF(Benchmark):

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
            "validation": "val",
            "test": "val",
        }

        self.dataset = self.filtered_hf_dataset(
            path="microsoft/MMLU-CF",
            name="default",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a multiple-choice question into a `Sample` object.
        """

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following multiple choice question.

            {record["Question"]}
        """
        ).strip()

        choices = [record["A"], record["B"], record["C"], record["D"]]

        # Append the choices, labeling each with a letter starting at 'A'
        choices_prompt = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
        )
        # print("choices_prompt", choices_prompt)

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n{choices_prompt}\n\n"
        output_format = f"Provide your final answer as a single letter in the range A-{chr(65 + len(choices) - 1)}."
        prompt += f"OUTPUT ANSWER FORMAT: {output_format}"

        # Determine the correct answer letter
        correct_answer_letter = record["Answer"]

        return Sample(
            input=prompt,
            target=correct_answer_letter,
            metadata={
                "format": output_format,
                "unique_id": record["unique_id"],
            },
        )

    @task
    def match_task(self):
        return Task(
            time_limit=self.args.task_timeout,
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.multi_choice_match(),
            config=GenerateConfig(temperature=0.5),
        )
