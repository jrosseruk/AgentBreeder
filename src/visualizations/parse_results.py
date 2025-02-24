import argparse
import json


def convert_ci_to_margin(data):
    """
    Converts confidence intervals from upper/lower bounds to median ± margin format.
    """
    for i, entry in enumerate(data):

        if "scaffold_safety_ci_median" in entry and entry.get(
            "scaffold_safety_ci_upper"
        ):

            safety_margin = entry.get("scaffold_safety_ci_upper", 0.0) - entry.get(
                "scaffold_safety_ci_median", 0.0
            )
            entry["scaffold_safety_ci"] = (
                f"{entry.get('scaffold_safety_ci_median', 0)*100:.1f} $\pm$ {safety_margin*100:.1f}"
            )
            # Store the median for comparison
            entry["_scaffold_safety_median"] = entry.get("scaffold_safety_ci_median", 0)

        if "scaffold_capability_ci_median" in entry and entry.get(
            "scaffold_capability_ci_upper"
        ):
            capability_margin = (
                entry["scaffold_capability_ci_upper"]
                - entry["scaffold_capability_ci_median"]
            )
            entry["scaffold_capability_ci"] = (
                f"{entry['scaffold_capability_ci_median']*100:.1f} $\pm$ {capability_margin*100:.1f}"
            )
            # Store the median for comparison
            entry["_scaffold_capability_median"] = entry.get(
                "scaffold_capability_ci_median", 0
            )

        if "scaffold_truth_ci_median" in entry and entry.get("scaffold_truth_ci_upper"):
            truth_margin = (
                entry["scaffold_truth_ci_upper"] - entry["scaffold_truth_ci_median"]
            )
            entry["scaffold_truth_ci"] = (
                f"{entry['scaffold_truth_ci_median']*100:.1f} $\pm$ {truth_margin*100:.1f}"
            )
            # Store the median for comparison
            entry["_scaffold_truth_median"] = entry.get("scaffold_truth_ci_median", 0)

    return data


def process_jsonl(input_file):
    """
    Reads a JSONL file, converts confidence intervals, identifies key scaffolds,
    and prints the modified data with annotations.
    """
    data = []
    # Read and process all entries
    with open(input_file, "r") as infile:
        for line in infile:
            entry = json.loads(line.strip())
            converted_entry = convert_ci_to_margin([entry])[0]
            # if float(converted_entry["scaffold_capability_ci_median"]) <= 0.1:
            #     print(converted_entry["scaffold_capability_ci_median"])

            #     continue
            data.append(converted_entry)

    # print(data[0])
    # data.sort(key=lambda x: x["scaffold_timestamp"])

    if not data:
        print("No data found in the input file.")
        return

    # Determine the most capable and most safe scaffolds
    max_capability = max(entry.get("_scaffold_capability_median", 0) for entry in data)
    max_safety = max(entry.get("_scaffold_safety_median", 0) for entry in data)
    max_truth = max(entry.get("_scaffold_truth_median", 0) for entry in data)

    # Identify Pareto optimal scaffolds
    pareto_optimal = []
    for entry in data:
        dominated = False
        for other in data:
            if (
                other.get("_scaffold_capability_median", 0)
                > entry.get("_scaffold_capability_median", 0)
                and other.get("_scaffold_safety_median", 0)
                >= entry.get("_scaffold_safety_median", 0)
            ) or (
                other.get("_scaffold_capability_median", 0)
                >= entry.get("_scaffold_capability_median", 0)
                and other.get("_scaffold_safety_median", 0)
                > entry.get("_scaffold_safety_median", 0)
            ):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(entry)

    # Print the results with annotations
    for result in data:
        annotations = []
        if result.get("_scaffold_capability_median", 0) == max_capability:
            annotations.append("(Most Capable)")
        if result.get("_scaffold_safety_median", 0) == max_safety:
            annotations.append("(Most Safe)")
        if result in pareto_optimal:
            annotations.append("(Pareto Optimal)")

        annotation_str = " ".join(annotations)
        print(
            f"{result['scaffold_name']} | {result.get('scaffold_capability_ci', 'NONE')} & {result.get('scaffold_safety_ci', 'NONE')} & {result.get('scaffold_truth_ci', 'NONE')} {annotation_str}"
        )

    return data


if __name__ == "__main__":

    input_file = "/home/#/Documents/AgentBreeder/src/results/20250129-234909/drop-36658085-9ce3-4b18-98ce-2fb8a4eec1b1.jsonl"

    process_jsonl(input_file)
