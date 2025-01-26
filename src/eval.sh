#!/bin/bash

# Define the list of popmarks
population_ids=(
    # "fd0ce977-dcce-4315-b293-1091b6908f41"
    "7f998181-98d9-446a-874a-ec12211e58d2"
    # "23bc8f1c-9c6a-43fd-8747-9e7493ce30f8"
    # "1de7ccb6-1a4e-45be-b68c-6862227df815"
)

echo "Launching evaluation for the following population_ids: ${population_ids[@]}"
# Path to your Python script
SCRIPT_PATH="src/eval.py"  # <-- Update this path

# Optional: Activate a virtual environment if needed
# source /path/to/your/venv/bin/activate

# Iterate over each popmark and launch it in a new terminal
for pop in "${population_ids[@]}"; do
    echo "Launching terminal for population_id: $pop"
    gnome-terminal -- bash -c "echo Running population_id: $pop; python3 \"$SCRIPT_PATH\" --population_id \"$pop\"; echo PopulationId '$pop' completed. Press Enter to close.; read"
done
