import argparse

import logging

from base import initialize_session, Population, System

import os
import uuid
import asyncio
import json
import warnings
from sqlalchemy.exc import SAWarning

import time
from evals import (
    CLRSText,
    MMLU,
    ARC,
    GPQA,
    DROP,
    MGSM,
    SaladData,
    SimpleQA,
    Math500,
    Math,
    MMLUCF,
    AntiSaladData,
)
import pandas as pd
import numpy as np
from rich import print

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


class Evaluator:

    def __init__(self, args):

        self.args = args

        self.benchmarks = {
            "arc": ARC,
            "gpqa": GPQA,
            "mmlu": MMLU,
            "drop": DROP,
            "mgsm": MGSM,
            "clrs_text": CLRSText,
            "salad_data": SaladData,
            "anti_salad_data": AntiSaladData,
            "simple_qa": SimpleQA,
            "math_500": Math500,
            "math": Math,
            "mmlu_cf": MMLUCF,
        }

        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args, split="test", shuffle=False, limit=self.args.n_evals
        )

    def evaluate(
        self,
        population: Population,
        systems_for_validation: list[System],
    ):

        model_metrics = self.benchmark.evaluate(
            systems_for_validation,
            limit=self.args.n_evals,
        )

        records = {
            system.system_id: {
                "population_id": population.population_id,
                "benchmark": population.population_benchmark,
                "system_name": system.system_name,
                "system_id": system.system_id,
                "system_safety_ci_median": None,
                "system_safety_ci_upper": None,
                "system_safety_ci_lower": None,
                "system_capability_ci_median": None,
                "system_capability_ci_upper": None,
                "system_capability_ci_lower": None,
                "system_code": system.system_code,
            }
            for system in systems_for_validation
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

                for system in systems_for_validation:

                    if str(system.system_id) == model.split("||")[1]:
                        if task == self.benchmarks[self.args.benchmark].__name__:
                            records[system.system_id]["system_capability_ci_median"] = (
                                metrics["median"]
                            )
                            records[system.system_id]["system_capability_ci_upper"] = (
                                metrics["ci_upper"]
                            )
                            records[system.system_id]["system_capability_ci_lower"] = (
                                metrics["ci_lower"]
                            )

                        elif task == SaladData.__name__:
                            records[system.system_id]["system_safety_ci_median"] = (
                                metrics["median"]
                            )
                            records[system.system_id]["system_safety_ci_upper"] = (
                                metrics["ci_upper"]
                            )
                            records[system.system_id]["system_safety_ci_lower"] = (
                                metrics["ci_lower"]
                            )

        return records


def compute_pareto_frontier(systems_in_cluster):
    def dominates(s1, s2):
        """
        Returns True if s1 dominates s2 across the two objectives:
        - system_capability_ci_median
        - system_safety_ci_median
        We assume we are maximizing both objectives.
        """
        if not s1.system_capability_ci_median:
            return False
        elif not s2.system_capability_ci_median:
            return True
        elif not s1.system_safety_ci_median:
            return True
        elif not s2.system_safety_ci_median:
            return False

        return (
            s1.system_capability_ci_median >= s2.system_capability_ci_median
            and s1.system_safety_ci_median >= s2.system_safety_ci_median
            and (
                s1.system_capability_ci_median > s2.system_capability_ci_median
                or s1.system_safety_ci_median > s2.system_safety_ci_median
            )
        )

    # Compute the Pareto front (non-dominated set)
    pareto_front = []
    for s1 in systems_in_cluster:
        # Check if s1 is dominated by any other system
        is_dominated = False
        for s2 in systems_in_cluster:
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
    parser.add_argument("--mode", type=str, default="blue")
    parser.add_argument("--n_evals", type=int, default=500)
    parser.add_argument("--task_timeout", type=int, default=60 * 60)
    parser.add_argument("--population_id", type=str, default=None)

    args = parser.parse_args()

    eval_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))

    for session in initialize_session():

        population = (
            session.query(Population).filter_by(population_id=args.population_id).one()
        )

        print(f"Evaluating population: {population.population_id}")

        args.benchmark = population.population_benchmark

        # args.benchmark = "gpqa"

        systems = (
            session.query(System)
            .filter_by(population_id=population.population_id)
            .all()
        )
        if len(systems) < 10:
            continue

        filtered_systems = []
        for system in systems:
            if (
                not system.system_capability_ci_median
                or system.system_capability_ci_median == 0
            ):
                continue  # Skip systems with median capability of 0
            filtered_systems.append(system)

        print(f"Found {len(filtered_systems)} systems for evaluation.")

        baseline_systems = filtered_systems[:7]

        generated_systems = filtered_systems[7:]
        generated_highest_capability_systems: list = sorted(
            generated_systems,
            key=lambda x: x.system_capability_ci_median,
            reverse=True,
        )[:10]

        if args.mode in ["blue", "red"]:
            generated_highest_safety_systems: list = sorted(
                generated_systems,
                key=lambda x: x.system_safety_ci_median,
                reverse=True,
            )[:10]
        else:
            generated_highest_safety_systems = []

        if args.mode in ["ablation"]:
            generated_highest_pareto_systems = []

        else:

            generated_highest_pareto_systems: list = compute_pareto_frontier(
                generated_systems
            )

        # eval all our systems
        systems_for_evaluation = (
            baseline_systems
            + generated_highest_capability_systems
            + generated_highest_safety_systems
            + generated_highest_pareto_systems
        )

        for system in systems:
            print(
                f"System: {system.system_name} | Capability: {system.system_capability_ci_median} | Safety: {system.system_safety_ci_median}"
            )

        time.sleep(10)

        # reverse them
        systems_for_evaluation = systems_for_evaluation[::-1]

        evaluator = Evaluator(args)
        records = evaluator.evaluate(population, systems_for_evaluation)

        results_file = f"./src/results/{eval_timestamp_str}/{population.population_benchmark}-{population.population_id}.jsonl"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "a") as f:
            for id, record in records.items():
                json_record = json.dumps(record)
                f.write(json_record + "\n")
