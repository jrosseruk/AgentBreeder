from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from evals.benchmark import Benchmark, register_benchmark


@register_benchmark("mgsm")
class MGSM(Benchmark):

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
            path="juletxara/mgsm",
            name=["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"],
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following maths question:

            {record["question"]}
        """
        ).strip()

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n\n"
        output_format = "Provide your final answer as a single number."
        prompt += f"OUTPUT ANSWER FORMAT: {output_format}"

        return Sample(
            input=prompt,
            target=str(record["answer_number"]),
            metadata={
                "format": output_format,
                "answer": record["answer"],
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
            scorer=self.llm_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @classmethod
    @property
    def prompt(self):
        return dedent(
            """
        Your aim is to find an optimal multi-agent scaffold performing well on the Multilingual Grade School Math Benchmark
        (MGSM) which evaluates mathematical problem-solving abilities across various languages to ensure
        broad and effective multilingual performance.
        ## An example question from MGSM:
        **Question**: この数学の問題を解いてください。
        近所では、ペットのウサギの数がペットの犬と猫を合わせた数よりも12匹少ない。犬1匹あたり2匹
        の猫がおり、犬の数は60匹だとすると、全部で近所には何匹のペットがいますか?
        **Answer (Not Given)**: 348
        """
        )
