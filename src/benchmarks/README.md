# Guide: Implementing a New Benchmark in the Framework

## Overview
This guide explains how to create a new benchmark class in the benchmark framework. The framework is designed to evaluate model performance on different tasks using a consistent interface.

## Step 1: Create a New Benchmark File

1. Create a new Python file in the `src/benchmarks` directory
2. Name it appropriately for your benchmark (e.g., `my_benchmark.py`)
3. Import required dependencies:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, accuracy, scorer
from evals.metrics import ci_lower, ci_upper, median
from benchmarks.benchmark import Benchmark, register_benchmark
from typing import Any, Literal, Union
```

## Step 2: Define the Benchmark Class

1. Create your benchmark class inheriting from `Benchmark`
2. Use the `@register_benchmark` decorator to register it:

```python
@register_benchmark("my_benchmark")
class MyBenchmark(Benchmark):
    def __init__(
        self,
        args: Any = None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ):
        self.args = args
        self.split = split

        # Define split mapping for dataset loading
        split_mapping = {
            "validation": "train",  # or appropriate split name
            "test": "test",
        }

        # Load and process the dataset
        self.dataset = self.filtered_hf_dataset(
            path="dataset/path",  # HuggingFace dataset path
            name="default",       # dataset configuration name
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed if self.args else None,
            limit=limit,
        )
```

## Step 3: Implement Required Methods

### 1. Record to Sample Conversion
Implement the `_record_to_sample` method to convert dataset records into `Sample` objects:

```python
def _record_to_sample(self, record: dict[str, Any]) -> Sample:
    """Convert a dataset record into a Sample object."""
    
    # Format the input prompt
    prompt = f"Your formatted prompt here: {record['question']}"
    
    # Define output format
    output_format = "Specify the expected output format"
    prompt += f"\nOUTPUT ANSWER FORMAT: {output_format}"
    
    return Sample(
        input=prompt,
        target=str(record["answer"]),
        metadata={
            "format": output_format,
            "unique_id": record["unique_id"],
            # Add other relevant metadata
        }
    )
```

### 2. Task Definition
Implement the `match_task` method to define how the benchmark task should be evaluated:

```python
@task
def match_task(self):
    """Define the evaluation task configuration."""
    return Task(
        time_limit=self.args.task_timeout,
        name=self.__class__.__name__,
        dataset=self.dataset,
        solver=self.match_solver(),
        scorer=self.your_scorer(),  # Define appropriate scorer
        config=GenerateConfig(temperature=0.5),
    )
```

### 3. Custom Scorer (Optional)
If needed, implement a custom scorer for your benchmark:

```python
@staticmethod
@scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
def your_scorer():
    async def score(state, target):
        try:
            # Implement scoring logic
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="your_scorer",
                    value=0,
                    answer=state.output.completion,
                    explanation="Error in model response",
                )
            
            # Your scoring logic here
            accuracy_value = compute_accuracy(state.output.completion, target.text)
            
            return Score(
                name="your_scorer",
                value=accuracy_value,
                answer=state.output.completion,
                explanation="Your explanation here",
            )
        except Exception as e:
            return Score(
                name="your_scorer",
                value=0,
                answer=state.output.completion,
                explanation=str(e),
            )
    return score
```

## Step 4: Optional Features

### 1. Dataset Filtering
Implement `benchmark_filter` if you need to filter dataset examples:

```python
def benchmark_filter(self, example):
    """
    Filter dataset examples based on specific criteria.
    Return True to keep the example, False to filter it out.
    """
    if some_condition(example):
        return True
    return False
```

### 2. Custom Metrics
Add custom metrics or evaluation logic as needed in your scorer.

## Step 5: Testing

1. Create a test file in `src/benchmarks/tests/test_my_benchmark.py`
2. Include basic tests for your benchmark:

```python
import unittest
from benchmarks.my_benchmark import MyBenchmark

class TestMyBenchmark(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--random_seed", type=int, default=42)
        self.args = parser.parse_args()
        
    def test_record_to_sample(self):
        evaluator = MyBenchmark(args=self.args, split="validation", limit=1)
        # Add your tests here
```

## Common Pitfalls to Avoid

1. **Dataset Loading**: Ensure your dataset path and configuration are correct
2. **Error Handling**: Implement proper error handling in scorers
3. **Memory Management**: Be mindful of dataset size and filtering
4. **Output Format**: Clearly specify expected output format in prompts
5. **Metrics**: Ensure your scoring metrics align with the task objectives

## Example Usage

After implementation, your benchmark can be used like this:

```python
benchmark = MyBenchmark(args=args, split="validation")
results = benchmark.evaluate(scaffolds, limit=100)
```