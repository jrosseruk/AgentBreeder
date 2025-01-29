import argparse
import json


def convert_ci_to_margin(data):
    """
    Converts confidence intervals from upper/lower bounds to median ± margin format.
    """
    for i, entry in enumerate(data):

        if "system_safety_ci_median" in entry and entry.get("system_safety_ci_upper"):

            safety_margin = entry.get("system_safety_ci_upper", 0.0) - entry.get(
                "system_safety_ci_median", 0.0
            )
            entry["system_safety_ci"] = (
                f"{entry.get('system_safety_ci_median', 0)*100:.1f} $\pm$ {safety_margin*100:.1f}"
            )
            # Store the median for comparison
            entry["_system_safety_median"] = entry.get("system_safety_ci_median", 0)

        if "system_capability_ci_median" in entry:
            capability_margin = (
                entry["system_capability_ci_upper"]
                - entry["system_capability_ci_median"]
            )
            entry["system_capability_ci"] = (
                f"{entry['system_capability_ci_median']*100:.1f} $\pm$ {capability_margin*100:.1f}"
            )
            # Store the median for comparison
            entry["_system_capability_median"] = entry["system_capability_ci_median"]

    return data


def process_jsonl(input_file):
    """
    Reads a JSONL file, converts confidence intervals, identifies key systems,
    and prints the modified data with annotations.
    """
    data = []
    # Read and process all entries
    with open(input_file, "r") as infile:
        for line in infile:
            entry = json.loads(line.strip())
            converted_entry = convert_ci_to_margin([entry])[0]
            # if float(converted_entry["system_capability_ci_median"]) <= 0.1:
            #     print(converted_entry["system_capability_ci_median"])

            #     continue
            data.append(converted_entry)

    # print(data[0])
    # data.sort(key=lambda x: x["system_timestamp"])

    if not data:
        print("No data found in the input file.")
        return

    # Determine the most capable and most safe systems
    max_capability = max(entry["_system_capability_median"] for entry in data)
    max_safety = max(entry.get("_system_safety_median", 0) for entry in data)

    # Identify Pareto optimal systems
    pareto_optimal = []
    for entry in data:
        dominated = False
        for other in data:
            if (
                other["_system_capability_median"] > entry["_system_capability_median"]
                and other.get("_system_safety_median", 0)
                >= entry.get("_system_safety_median", 0)
            ) or (
                other["_system_capability_median"] >= entry["_system_capability_median"]
                and other.get("_system_safety_median", 0)
                > entry.get("_system_safety_median", 0)
            ):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(entry)

    # Print the results with annotations
    for result in data:
        annotations = []
        if result["_system_capability_median"] == max_capability:
            annotations.append("(Most Capable)")
        if result.get("_system_safety_median", 0) == max_safety:
            annotations.append("(Most Safe)")
        if result in pareto_optimal:
            annotations.append("(Pareto Optimal)")

        annotation_str = " ".join(annotations)
        print(
            f"{result['system_name']} | {result['system_capability_ci']} & {result.get('system_safety_ci', 'NONE')} {annotation_str}"
        )

    return data


if __name__ == "__main__":

    input_file = "/home/j/Documents/AgentBreeder/src/results/20250129-083111/gpqa-13e8fb8c-4d16-4487-85ce-e40249da8422.jsonl"

    process_jsonl(input_file)
