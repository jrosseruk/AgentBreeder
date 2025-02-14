import ast
import json
import sys

sys.path.append("src")
import time
import os
from base import Scaffold
from evals import benchmark_registry


def extract_class_code(file_path: str, class_name: str) -> str:
    """
    Extracts the code for a specified class from a Python file.

    Args:
        file_path (str): Path to the Python file.
        class_name (str): Name of the class to extract.

    Returns:
        str: The code of the specified class as a string, or None if not found.
    """
    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        # Parse the file content into an AST
        tree = ast.parse(file_content)

        # Locate the specified class in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract the source code for the class
                start_line = node.lineno - 1
                end_line = max(
                    [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")]
                )
                return "\n".join(file_content.splitlines()[start_line:end_line])

        return None  # Class not found

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_function_code(file_path: str, function_name: str) -> str:
    """
    Extracts the code for a specified function from a Python file, including any decorators.

    Args:
        file_path (str): Path to the Python file.
        function_name (str): Name of the function to extract.

    Returns:
        str: The code of the specified function (including decorators) as a string, or None if not found.
    """
    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        # Parse the file content into an AST
        tree = ast.parse(file_content)

        # Locate the specified function in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Determine the start and end lines, including decorators
                decorators = node.decorator_list
                start_line = (
                    min(
                        (decorator.lineno for decorator in decorators),
                        default=node.lineno,
                    )
                    - 1
                )
                end_line = max(
                    [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")]
                )

                # Return the source code for the function with decorators
                return "\n".join(file_content.splitlines()[start_line:end_line])

        return None  # Function not found

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_base_prompt_with_archive(args, session):
    from .meta_agent_prompts import EXAMPLE, BASE_PROMPT

    current_directory = os.path.dirname(os.path.abspath(__file__))
    higher_directory = os.path.dirname(current_directory)

    scaffold_agent_code = extract_class_code(
        f"{higher_directory}/base/scaffold.py", "Agent"
    )

    BASE_PROMPT = BASE_PROMPT.replace("[SCAFFOLD_CODE]", scaffold_agent_code)

    # Only choose scaffolds which haven't been validated yet (e.g. scaffold_fitness=None)
    init_archive = (
        session.query(Scaffold)
        .filter_by(population_id=args.population_id, scaffold_first_parent_id=None)
        .all()
    )

    archive = (
        session.query(Scaffold)
        .filter_by(population_id=args.population_id)
        .order_by(Scaffold.scaffold_fitness.desc())
        .all()[:10]
    )

    # reverse the archive list to have the best solutions last
    archive = archive[::-1]

    archive_jsonl = []
    for scaffold in init_archive + archive:
        if scaffold.scaffold_name not in [a["name"] for a in archive_jsonl]:
            archive_jsonl.append(
                {
                    "thought": scaffold.scaffold_thought_process,
                    "name": scaffold.scaffold_name,
                    "code": scaffold.scaffold_agent_code,
                    "fitness": str(scaffold.scaffold_capability_ci_median),
                    "safety": str(scaffold.scaffold_safety_ci_median),
                }
            )

    archive_str = ",\n".join([json.dumps(sol, indent=4) for sol in archive_jsonl])
    archive_str = f"[{archive_str}]"
    prompt = BASE_PROMPT.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    prompt = prompt.replace(
        "[BENCHMARK_DESCRIPTION]", str(benchmark_registry[args.benchmark].prompt)
    )
    if args.mode in ["blue"]:
        prompt = prompt.replace(
            "[SAFETY_DESCRIPTION]", str(benchmark_registry["salad_data"].prompt)
        )
    elif args.mode in ["red"]:
        prompt = prompt.replace(
            "[SAFETY_DESCRIPTION]", str(benchmark_registry["anti_salad_data"].prompt)
        )

    else:
        prompt = prompt.replace("[SAFETY_DESCRIPTION]", "")

    response_format = {
        "thought": "The first key should be (thought), and it should capture your thought process for designing the next function. In the thought section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.",
        "name": "The second key (name) corresponds to the name of your next agent architecture.",
        "code": "Finally, the last key (code) corresponds to the exact forward() function in Python code that you would like to try. You must write a COMPLETE CODE in code: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.",
    }

    return prompt, response_format
