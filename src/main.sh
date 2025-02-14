#!/bin/bash

# Define the list of benchmarks
benchmarks=(
    "gpqa"
    # "drop"
    # "math"
    # "mmlu"
    # "mmlu_cf"
    # "arc"
    # "simple_qa"
    # "clrs_text"
    # "salad_data"
    # "anti_salad_data"
    # "truthful_qa"
    # "mgsm" #saturated
)

# Define your mode, blue, red or ablation
MODE="blue"

# Path to your Python script
SCRIPT_PATH="src/main.py"  # <-- Update this path


# Optional: Activate a virtual environment if needed
# source /path/to/your/venv/bin/activate


# Start the API routers
gnome-terminal -- bash -c "echo Starting up openai router: $bench; python3 src/api/openai_api.py ; read"
gnome-terminal -- bash -c "echo Starting up anthropic router: $bench; python3 src/api/anthropic_api.py ; read"


# Iterate over each benchmark and launch it in a new terminal
for bench in "${benchmarks[@]}"; do
    gnome-terminal -- bash -c "echo Running benchmark: $bench; python3 \"$SCRIPT_PATH\" --benchmark \"$bench\" --mode \"$MODE\"; echo Benchmark '$bench' completed. Press Enter to close.; read"
done
