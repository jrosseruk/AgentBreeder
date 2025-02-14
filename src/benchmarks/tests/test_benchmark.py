import sys

sys.path.append("src")
import argparse

from benchmarks.math_500 import Math500
from evals.negative_sampler import get_positive_and_negative_samples

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=42)
args = parser.parse_args()

m = Math500(args, split="validation", shuffle=True, limit=1000)

# print(m.dataset)


p_n = get_positive_and_negative_samples("Math500")

print(p_n[0])


# for record in m.dataset:
#     print(record.metadata["unique_id"])


p = 0
n = 0
u = 0

for record in m.dataset:
    if record.metadata["unique_id"] in p_n[1.0]:
        p += 1
    if record.metadata["unique_id"] in p_n[0]:
        n += 1
    if (
        record.metadata["unique_id"] not in p_n[1.0]
        and record.metadata["unique_id"] not in p_n[0]
    ):
        u += 1

print("p", p, "n", n, "u", u)
print(p + n + u)
