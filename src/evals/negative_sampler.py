import json
from collections import defaultdict
import os
import glob
from datetime import datetime
from rich import print


def find_all_baseline_files(validation_dir, Benchmark):
    """
    Finds all '{Benchmark}' JSON files in the logs directories under all '{Benchmark}-<id>'
    directories within the latest timestamped subdirectory in the validation directory.

    :param validation_dir: Path to the 'validation' directory.
    :return: List of full paths to all '{Benchmark}' JSON files or an empty list if none found.
    """
    # Step 1: List all timestamped subdirectories
    timestamp_dirs = [
        d
        for d in os.listdir(validation_dir)
        if os.path.isdir(os.path.join(validation_dir, d))
        and d.startswith("20")  # Assuming timestamps start with '20'
    ]

    if not timestamp_dirs:
        print("No timestamped directories found.")
        return []

    # Step 2: Sort the directories to find the latest one
    # Assuming the format is YYYYMMDD-HHMMSS
    try:
        timestamp_dirs_sorted = sorted(
            timestamp_dirs,
            key=lambda x: datetime.strptime(x, "%Y%m%d-%H%M%S"),
            reverse=True,
        )
    except ValueError as e:
        print(f"Timestamp format error: {e}")
        return []

    for latest_timestamp_dir in timestamp_dirs_sorted:

        latest_timestamp_path = os.path.join(validation_dir, latest_timestamp_dir)
        print(f"Latest timestamp directory: {latest_timestamp_dir}")

        math_dirs = [
            d
            for d in os.listdir(latest_timestamp_path)
            if os.path.isdir(os.path.join(latest_timestamp_path, d))
            and d.startswith(f"{Benchmark}-")
        ]
        print(f"Found {len(math_dirs)} '{Benchmark}-' directories.")

        if len(math_dirs) == 0:
            print(
                f"No '{Benchmark}-' directories found in the latest timestamp directory."
            )

        else:
            print(f"Found {len(math_dirs)} '{Benchmark}-' directories.")
            break

    # Step 4: Collect all JSON files in each {Benchmark}-<id>/logs/ directory
    all_math_files = []
    for math_dir in math_dirs:
        logs_path = os.path.join(latest_timestamp_path, math_dir, "logs")
        if not os.path.exists(logs_path):
            print(f"'logs' directory does not exist in {math_dir}. Skipping.")
            continue

        # Pattern to match JSON files containing '{Benchmark}'
        pattern = os.path.join(logs_path, f"*_{Benchmark}_*.json")
        found_files = glob.glob(pattern)
        if found_files:
            print(
                f"Found {len(found_files)} '{Benchmark}' JSON files in {math_dir}/logs/"
            )
        else:
            print(f"No '{Benchmark}' JSON files found in {math_dir}/logs/")
        all_math_files.extend(found_files)

    if not all_math_files:
        print(f"No '{Benchmark}' JSON files found in any 'logs' directories.")
        return []

    print(f"Total '{Benchmark}' JSON files found: {len(all_math_files)}")
    return all_math_files


def create_score_to_unique_ids_dict(json_str):
    """
    Parses the JSON string and creates a dictionary mapping scores to unique IDs.

    :param json_str: JSON string containing the data.
    :return: Dictionary with scores as keys and lists of unique IDs as values.
    """
    # Parse the JSON data
    data = json.loads(json_str)

    # Initialize a defaultdict of lists
    score_to_ids = defaultdict(list)

    # Iterate over each sample
    for sample in data.get("samples", []):
        # Extract the score
        _, score_info = list(sample.get("scores", {}).items())[0]
        score = score_info.get("value")

        # Ensure the score is a float
        if score is not None:
            try:
                score = float(score)
            except ValueError:
                continue  # Skip if score is not a valid float
        else:
            continue  # Skip if score is missing

        # Extract the unique_id
        unique_id = sample.get("metadata", {}).get("unique_id")
        if unique_id:
            score_to_ids[score].append(unique_id)

    # Convert defaultdict to regular dict
    score_to_ids = dict(score_to_ids)

    return score_to_ids


def get_positive_and_negative_samples(Benchmark="GPQA"):
    validation_directory = "/home/#/Documents/AgentBreeder/src/baselines/validation/"
    try:
        all_math_files = find_all_baseline_files(validation_directory, f"{Benchmark}")

        if all_math_files:
            print(f"List of all '{Benchmark}' JSON files:")

            scores = []
            for file_path in all_math_files:
                print(file_path)

                with open(file_path, "r") as f:
                    f = f.read()

                    score_to_ids = create_score_to_unique_ids_dict(f)

                scores.append(score_to_ids)

            # print(scores)

            # Work out best item in score_to_ids
            totals = []
            for s in scores:
                running_total = 0
                for k, v in s.items():
                    running_total += k * len(v)
                totals.append(running_total)

            # print(totals)

            best_idx = totals.index(max(totals))

            best = scores[best_idx]

            # ensure that each item in best.values() is unique. if the same item appears across 2 keys, choosse the key where it appears the most to keep and remove it from the other one

            # Step 1: Count occurrences of each item per key
            item_counts = defaultdict(lambda: defaultdict(int))
            for key, items in best.items():
                for item in items:
                    item_counts[item][key] += 1

            # Step 2: Determine the best key for each item
            item_best_key = {}
            for item, counts in item_counts.items():
                # Find the maximum count
                max_count = max(counts.values())
                # Find all keys with the maximum count
                best_keys = [key for key, count in counts.items() if count == max_count]
                # Select the lowest key in case of a tie
                best_key = min(best_keys)
                item_best_key[item] = best_key

            # Step 3: Assign items uniquely to their best keys
            best_dict = defaultdict(list)
            for item, key in item_best_key.items():
                best_dict[key].append(item)

            # Optional: Sort the keys for better readability
            best = dict(sorted(best_dict.items()))

            # print(f"Best: {best}")

        else:
            print(f"No '{Benchmark}' JSON files found.")
            return {}

        for key, value in best.items():

            if key != 1.0 and key != 0.0:
                best = split_dict_entries(best)
                break
        return best
    except:
        return {}


def split_dict_entries(data):
    # Flatten the dictionary to get all strings and their associated values
    all_strings = []
    for key, values in data.items():
        all_strings.extend((key, val) for val in values)

    # Sort strings by their original keys (ascending order)
    all_strings.sort(reverse=True, key=lambda x: x[0])

    # Calculate the number of strings to assign to the positive sample (1.0)
    total_strings = len(all_strings)
    positive_count = total_strings // 2

    # Distribute strings into positive (1.0) and negative (0.0) buckets
    positive_bucket = [val for _, val in all_strings[:positive_count]]
    negative_bucket = [val for _, val in all_strings[positive_count:]]

    # Construct the resulting dictionary
    result = {1.0: positive_bucket, 0.0: negative_bucket}
    return result


# Example usage
if __name__ == "__main__":
    samples = get_positive_and_negative_samples("DROP")
    print(samples)
