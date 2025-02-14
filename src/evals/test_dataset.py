import sys

sys.path.append("src")

from base import System
import unittest
from evals.benchmarks.drop import DROP
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from prompts.initial_population import COT_SC
import re
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=42)

args = parser.parse_args()

evaluator = DROP(args=args, split="validation", limit=1)

print(evaluator.dataset[0])
