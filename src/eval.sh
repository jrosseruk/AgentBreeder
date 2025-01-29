#!/bin/bash

# Define the list of popmarks
population_ids=(
    "36594cd9-9902-4f69-9651-347e14eb1990"
    "7f998181-98d9-446a-874a-ec12211e58d2"
    "13e8fb8c-4d16-4487-85ce-e40249da8422"
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
