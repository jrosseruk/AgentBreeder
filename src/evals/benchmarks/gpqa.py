from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset

from typing import Any, Literal, Union
from textwrap import dedent
import random

from evals.benchmark import Benchmark, register_benchmark


@register_benchmark("gpqa")
class GPQA(Benchmark):

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
            "test": "train",
        }

        self.dataset = self.filtered_hf_dataset(
            path="Idavidrein/gpqa",
            name="gpqa_diamond",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

        print(f"Dataset size: {len(self.dataset)}")

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

        choices = [
            record["Correct Answer"],
            record["Incorrect Answer 1"],
            record["Incorrect Answer 2"],
            record["Incorrect Answer 3"],
        ]

        random.shuffle(choices)

        # Append the choices, labeling each with a letter starting at 'A'
        choices_prompt = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
        )
        # print("choices_prompt", choices_prompt)

        # Combine question and choices into a single prompt
        prompt = (
            f"{question_prompt}\n{choices_prompt}\n"  # Removed the extra line break
        )
        output_format = f"""Provide your final answer as a single letter in the range A-{chr(65 + len(choices) - 1)}."""
        prompt += f"OUTPUT ANSWER FORMAT: {output_format}"

        # Determine the correct answer letter
        correct_answer_letter = chr(65 + choices.index(record["Correct Answer"]))

        return Sample(
            input=prompt,
            target=correct_answer_letter,
            metadata={
                "format": output_format,
                "correct_answer": record["Correct Answer"],
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

    @classmethod
    @property
    def prompt(self):
        return dedent(
            """
        Your aim is to find an optimal multi-agent scaffold performing well on the GPQA (Graduate-Level Google-Proof Q&A
        Benchmark). This benchmark consists of challenging multiple-choice questions across the domains of
        biology, physics, and chemistry, designed by domain experts to ensure high quality and difficulty.
        ## An example question from GPQA:
        Two quantum states with energies E1 and E2 have a lifetime of 10-9 sec and 10-8 sec, respectively. We
        want to clearly distinguish these two energy levels. Which one of the following options could be their
        energy difference so that they be clearly resolved?
        Answer choices:
        10-9 eV
        10-8 eV
        10-7 eV
        10-6 eV
        Correct answer [Not provided]:
        10-7 eV
        Explanation [Not provided]:
        According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the
        width of the energy level. With Delta t=10-9 s==> Delta E1= 3.3 10-7 ev. And Delta t=10-11 s gives
        Delta E2=3.310-8 eV. Therefore, the energy difference between the two states must be significantly
        greater than 10-7 ev. So the answer is 10-4 ev
        """
        )
