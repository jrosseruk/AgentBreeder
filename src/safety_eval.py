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
    AntiSaladData,
    TruthfulQA,
    SimpleQA,
    Math500,
)
import pandas as pd
import numpy as np
from rich import print
from evals.validator import Validator

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--log_timestamp", type=str, default=log_timestamp_str)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_generation", type=int, default=10)
    parser.add_argument("--n_mutations", type=int, default=10)
    parser.add_argument("--n_evals", type=int, default=100)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--mode", type=str, default="ablation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("-p", "--population_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="salad_data")
    parser.add_argument("--task_timeout", type=int, default=20 * 60)

    args = parser.parse_args()

    eval_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))

    # try:
    populations = []
    for session in initialize_session():

        population_ids = [
            "67ff394d-5589-48dc-83ed-c8209b347692",
            "ee7b2ebd-08bb-416c-a691-91dbabb5b7df",
            "52a12038-10b8-4887-b782-85e7d6330306",
        ]

        for population_id in population_ids:

            population = (
                session.query(Population).filter_by(population_id=population_id).one()
            )

            args.benchmark = population.population_benchmark

            print(population_id, population.population_benchmark)

            systems = session.query(System).filter_by(population_id=population_id).all()

            # do chunks of 20 systems at a time

            # reverse the systems list:
            systems = systems[::-1]

            for i in range(0, len(systems), 20):
                systems_chunk = systems[i : i + 20]

                validator = Validator(args)

                validator.validate(systems_chunk)
