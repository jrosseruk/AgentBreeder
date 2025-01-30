#!/bin/bash




# Define the list of popmarks
population_ids=(
    # "36658085-9ce3-4b18-98ce-2fb8a4eec1b1"
    # "cfda0d48-e4aa-439b-a348-b1433d27d344"
    "cecec343-5f63-4a02-99b8-7d0155d7c45f"
)

echo "Launching evaluation for the following population_ids: ${population_ids[@]}"
# Path to your Python script
SCRIPT_PATH="src/eval.py"  # <-- Update this path

# Optional: Activate a virtual environment if needed
# source /path/to/your/venv/bin/activate

# Iterate over each popmark and launch it in a new terminal
for pop in "${population_ids[@]}"; do
    echo "Launching terminal for population_id: $pop"
    # Sleep for 4 hours (14400 seconds)
    # sleep 14400
    gnome-terminal -- bash -c "echo Running population_id: $pop; python3 \"$SCRIPT_PATH\" --population_id \"$pop\"; echo PopulationId '$pop' completed. Press Enter to close.; read"
done
