import argparse

import logging

from base import initialize_session, Population, Scaffold

import os
import uuid
import asyncio
import json
import warnings
from sqlalchemy.exc import SAWarning

import time
from benchmarks.benchmark import benchmark_registry
from evals import (
    SaladData,
    AntiSaladData,
    TruthfulQA,
)
import pandas as pd
import numpy as np
from rich import print

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)
import sys


def extract_async_forward_as_string(file_path):
    """
    Extracts the async def forward method from the given Python file and returns it as a string.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        str: The extracted async def forward method as a string.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    forward_lines = []
    in_forward = False
    indent_level = None

    for line in lines:
        stripped_line = line.lstrip()

        # Check if the line starts the async def forward method
        if not in_forward and stripped_line.startswith("async def forward("):
            in_forward = True
            indent_level = len(line) - len(stripped_line)
            forward_lines.append(stripped_line.rstrip())
            continue

        if in_forward:
            current_indent = len(line) - len(stripped_line)

            # If we encounter a line with indentation less than or equal to the method definition, stop
            if stripped_line and current_indent <= indent_level:
                break

            # Remove the initial indentation
            if indent_level is not None:
                forward_lines.append(line[indent_level:].rstrip())
            else:
                forward_lines.append(line.rstrip())

    # Combine the lines into a single string
    forward_string = "\n".join(forward_lines)
    return forward_string


class Evaluator:

    def __init__(self, args):

        self.args = args

        self.benchmarks = benchmark_registry

        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args, split="test", shuffle=False, limit=self.args.n_evals
        )

    def evaluate(
        self,
        scaffold: Scaffold,
    ):

        model_metrics = self.benchmark.evaluate(
            [scaffold],
            limit=self.args.n_evals,
        )

        records = {
            scaffold.scaffold_id: {
                "scaffold_name": scaffold.scaffold_name,
                "scaffold_id": scaffold.scaffold_id,
                "scaffold_safety_ci_median": None,
                "scaffold_safety_ci_upper": None,
                "scaffold_safety_ci_lower": None,
                "scaffold_capability_ci_median": None,
                "scaffold_capability_ci_upper": None,
                "scaffold_capability_ci_lower": None,
                "scaffold_code": scaffold.scaffold_code,
            }
            for scaffold in [scaffold]
        }

        # You can now access each model’s metrics via model_metrics
        for model, task_metrics in model_metrics.items():
            for task, metrics in task_metrics.items():

                print(f"Model: {model}")
                print(f"Task: {task}")
                print(f"  accuracy: {metrics['accuracy']}")
                print(f"  ci_lower: {metrics['ci_lower']}")
                print(f"  ci_upper: {metrics['ci_upper']}")
                print(f"  median:   {metrics['median']}")

                for scaffold in [scaffold]:

                    if str(scaffold.scaffold_id) == model.split("||")[1]:
                        if task == self.benchmarks[self.args.benchmark].__name__:
                            records[scaffold.scaffold_id][
                                "scaffold_capability_ci_median"
                            ] = metrics["median"]
                            records[scaffold.scaffold_id][
                                "scaffold_capability_ci_upper"
                            ] = metrics["ci_upper"]
                            records[scaffold.scaffold_id][
                                "scaffold_capability_ci_lower"
                            ] = metrics["ci_lower"]

                        elif (
                            task == SaladData.__name__ or task == AntiSaladData.__name__
                        ):
                            records[scaffold.scaffold_id][
                                "scaffold_safety_ci_median"
                            ] = metrics["median"]
                            records[scaffold.scaffold_id][
                                "scaffold_safety_ci_upper"
                            ] = metrics["ci_upper"]
                            records[scaffold.scaffold_id][
                                "scaffold_safety_ci_lower"
                            ] = metrics["ci_lower"]

                        elif task == TruthfulQA.__name__:
                            records[scaffold.scaffold_id][
                                "scaffold_truth_ci_median"
                            ] = metrics["median"]
                            records[scaffold.scaffold_id]["scaffold_truth_ci_upper"] = (
                                metrics["ci_upper"]
                            )
                            records[scaffold.scaffold_id]["scaffold_truth_ci_lower"] = (
                                metrics["ci_lower"]
                            )

        return records


def compute_pareto_frontier(scaffolds_in_cluster):
    def dominates(s1, s2):
        """
        Returns True if s1 dominates s2 across the two objectives:
        - scaffold_capability_ci_median
        - scaffold_safety_ci_median
        We assume we are maximizing both objectives.
        """
        if not s1.scaffold_capability_ci_median:
            return False
        elif not s2.scaffold_capability_ci_median:
            return True
        elif not s1.scaffold_safety_ci_median:
            return True
        elif not s2.scaffold_safety_ci_median:
            return False

        return (
            s1.scaffold_capability_ci_median >= s2.scaffold_capability_ci_median
            and s1.scaffold_safety_ci_median >= s2.scaffold_safety_ci_median
            and (
                s1.scaffold_capability_ci_median > s2.scaffold_capability_ci_median
                or s1.scaffold_safety_ci_median > s2.scaffold_safety_ci_median
            )
        )

    # Compute the Pareto front (non-dominated set)
    pareto_front = []
    for s1 in scaffolds_in_cluster:
        # Check if s1 is dominated by any other scaffold
        is_dominated = False
        for s2 in scaffolds_in_cluster:
            if s1 == s2:
                continue
            if dominates(s2, s1):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(s1)

    return pareto_front


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--log_timestamp", type=str, default=log_timestamp_str)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--mode", type=str, default="ablation")
    parser.add_argument("--n_evals", type=int, default=500)
    parser.add_argument("--task_timeout", type=int, default=60 * 60)
    parser.add_argument("--population_id", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="gpqa")

    args = parser.parse_args()

    eval_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))

    for session in initialize_session():

        file_path = f"src/adas/{args.benchmark}.py"
        forward_string = extract_async_forward_as_string(file_path)
        assert forward_string and len(forward_string) > 0

        # reverse them
        scaffold = Scaffold(
            scaffold_id=str(uuid.uuid4()),
            scaffold_name=f"ADAS_{args.benchmark}",
            scaffold_code=forward_string,
        )

        evaluator = Evaluator(args)

        records = evaluator.evaluate(scaffold)

        results_file = f"./src/results/{eval_timestamp_str}/{args.benchmark}-{scaffold.scaffold_name}.jsonl"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "a") as f:
            for id, record in records.items():
                json_record = json.dumps(record)
                f.write(json_record + "\n")
