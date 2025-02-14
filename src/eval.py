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
from benchmarks.benchmark import benchmark_registry
from evals import (
    SaladData,
    TruthfulQA,
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

        self.benchmarks = benchmark_registry

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

                        elif (
                            task == SaladData.__name__ or task == AntiSaladData.__name__
                        ):
                            records[system.system_id]["system_safety_ci_median"] = (
                                metrics["median"]
                            )
                            records[system.system_id]["system_safety_ci_upper"] = (
                                metrics["ci_upper"]
                            )
                            records[system.system_id]["system_safety_ci_lower"] = (
                                metrics["ci_lower"]
                            )

                        elif task == TruthfulQA.__name__:
                            records[system.system_id]["system_truth_ci_median"] = (
                                metrics["median"]
                            )
                            records[system.system_id]["system_truth_ci_upper"] = (
                                metrics["ci_upper"]
                            )
                            records[system.system_id]["system_truth_ci_lower"] = (
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
    parser.add_argument("--mode", type=str, default="ablation")
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
            .order_by(System.generation_timestamp)
            .all()
        )
        if len(systems) < 10:
            continue

        filtered_systems = []
        generation_set = set()
        final_generation_systems = []
        for system in systems:
            generation_set.add(system.generation_timestamp)
            if (
                not system.system_capability_ci_median
                or system.system_capability_ci_median == 0
            ):
                continue  # Skip systems with median capability of 0
            elif len(generation_set) > 21:
                continue  # Skip systems after the 10th generation
            elif len(generation_set) == 21:
                final_generation_systems.append(system)

            filtered_systems.append(system)

        print(f"Found {len(filtered_systems)} systems for evaluation.")
        print("Number of generations", len(generation_set))

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
            )[:5]
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
            + final_generation_systems
        )

        for system in systems:
            print(
                f"System: {system.system_name} | Capability: {system.system_capability_ci_median} | Safety: {system.system_safety_ci_median}"
            )

        time.sleep(3)

        # reverse them
        systems_for_evaluation = list(set(systems_for_evaluation[::-1]))

        records = {}
        evaluator = Evaluator(args)
        # for i in range(0, len(systems_for_evaluation), 20):
        #     systems_chunk = systems_for_evaluation[i : i + 20]

        #     records.update(evaluator.evaluate(population, systems_chunk))

        records = evaluator.evaluate(population, systems_for_evaluation)

        results_file = f"./src/results/{eval_timestamp_str}/{population.population_benchmark}-{population.population_id}.jsonl"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "a") as f:
            for id, record in records.items():
                json_record = json.dumps(record)
                f.write(json_record + "\n")
