import json

# Load the JSON data from the file
with open(
    "/home/#/Documents/AgentBreeder/src/results/20250126-150029/mmlu-7f998181-98d9-446a-874a-ec12211e58d2.jsonl",
    "r",
) as file:
    json_data = [json.loads(line) for line in file]

# Sort the frameworks by system_capability_ci_median
sorted_frameworks = sorted(
    json_data, key=lambda x: x.get("system_capability_ci_median", 0), reverse=True
)

# Print the framework names in order
print("Frameworks sorted by system_capability_ci_median:")
for framework in sorted_frameworks:
    print(
        f"{framework['system_name']} - {framework.get('system_capability_ci_median', 'N/A')}"
    )
