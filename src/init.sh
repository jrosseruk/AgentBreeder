#!/bin/bash

# Define the list of benchmarks
benchmarks=(
    # "mmlu_cf"
    # "math_500"
    # "gpqa"
    # "drop"
    # "math"
    # "mmlu"
    # "arc"
    # "simple_qa"
    # "clrs_text"
    # "salad_data"
    # "anti_salad_data"
    "truthful_qa"
    # "mgsm" #saturated
)

# Path to your Python script
SCRIPT_PATH="src/initialize_baselines.py"  # <-- Update this path

# Optional: Activate a virtual environment if needed
# source /path/to/your/venv/bin/activate

# Iterate over each benchmark and launch it in a new terminal
for bench in "${benchmarks[@]}"; do
    gnome-terminal -- bash -c "echo Running benchmark: $bench; python3 \"$SCRIPT_PATH\" --benchmark \"$bench\"; echo Benchmark '$bench' completed. Press Enter to close.; read"
done
