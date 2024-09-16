import json
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path


# EXP_LOG_FILES = [
#     "veval/logs/2024-09-16T12-34-28-04-00_multihop-rag_NQHuaWYeQzGWCWVQxYLVgF.json",
#     "veval/logs/2024-09-16T12-52-49-04-00_multihop-rag_86TiNCLciSdb8in2a7hRpL.json",
#     "veval/logs/2024-09-16T13-44-37-04-00_multihop-rag_JTf8DqHbbDDRtXv5QAqZyD.json",
#     "veval/logs/2024-09-16T13-57-00-04-00_multihop-rag_3WKAK3B97eQasA83XJ4Mu7.json",
# ]
EXP_LOG_FILES = [
    "veval/logs/2024-09-09T10-05-49-04-00_multihop-rag_JijWWjj6BgP9U3gkyDdDPB.json",
    "veval/logs/2024-09-09T10-27-52-04-00_multihop-rag_2xfGcYrhJGQ3whxtjuF3wo.json",
    "veval/logs/2024-09-09T10-30-38-04-00_multihop-rag_QENVFpKcQGGsx8crJTtFtH.json",
    "veval/logs/2024-09-09T10-44-46-04-00_multihop-rag_n2bdvThkRSVU6gxRWiVMbn.json",
]


def extract_results(log_dict):
    using_rag = "w_rag" in log_dict["eval"]["task_file"]
    dataset = log_dict["eval"]["dataset"]
    model = log_dict["eval"]["model"]

    acc_dist = defaultdict(list)
    for sample in log_dict["samples"]:
        acc_dist[sample["metadata"]["evolution_type"]].append(
            sample["scores"]["veval/_scorer"]["value"]["answer_correctness"]
        )
    acc_dist = {
        k: {"acc_mean": np.mean(v), "samples": len(v)} for k, v in acc_dist.items()
    }

    return {
        "dataset": dataset,
        "model": model,
        "using_rag": using_rag,
        "metrics": {
            "acc_mean": log_dict["results"]["scores"][0]["metrics"][
                "answer_correctness/mean"
            ]["value"],
            "acc_std": log_dict["results"]["scores"][0]["metrics"][
                "answer_correctness/bootstrap_std"
            ]["value"],
            "acc_dist": acc_dist,
        },
    }


exp_results = []
for exp_file in EXP_LOG_FILES:
    with open(Path(__file__).parent.parent / exp_file, "r") as f:
        exp_log_dict = json.load(f)
    exp_results.append(extract_results(exp_log_dict))
# pprint(exp_results)


# Plot results
dataset_name = [
    exp_results[idx]["dataset"]["name"].split("/")[-1]
    for idx in range(len(exp_results))
]
dataset_name = np.unique(dataset_name)
assert len(dataset_name) == 1
dataset_name = dataset_name[0]

plot_data = defaultdict(dict)
for exp in exp_results:
    plot_data[exp["model"]][f"{('w' if exp['using_rag'] else 'w/o')} RAG"] = exp[
        "metrics"
    ]

x_ticks_lvl1 = ("w/o RAG", "w RAG")
x_ticks_lvl2 = ("openai/gpt-4-1106-preview", "openai/gpt-4-turbo-preview")
acc_dist_data = {
    q_type: [
        plot_data[x_ticks_lvl2[idx // 2]][x_ticks_lvl1[idx % 2]]["acc_dist"][q_type][
            "acc_mean"
        ]
        for idx in range(4)
    ]
    for q_type in ["simple", "multi_context", "reasoning"]
}
acc_data = [
    plot_data[x_ticks_lvl2[idx // 2]][x_ticks_lvl1[idx % 2]]["acc_mean"]
    for idx in range(4)
]
q_type_count = {
    k: v["samples"] for k, v in exp_results[0]["metrics"]["acc_dist"].items()
}

x = np.arange(len(x_ticks_lvl1) * len(x_ticks_lvl2))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in acc_dist_data.items():
    offset = width * multiplier
    rects = ax.bar(
        x + offset, measurement, width, label=f"{attribute} ({q_type_count[attribute]})"
    )
    ax.bar_label(rects, padding=2, fmt="%.2f", label_type="center")
    multiplier += 1

# line plots for overall acc
line1 = ((x + width)[:2], acc_data[:2])
line2 = ((x + width)[2:], acc_data[2:])
ax.plot(line1[0], line1[1], marker="o", color="black")
ax.plot(line2[0], line2[1], marker="o", color="black")

ax.set_ylabel("Mean Acc")
ax.set_title(dataset_name)
ax.set_xticks(x + width, x_ticks_lvl1 * 2)
sec_ax = ax.secondary_xaxis(location=0)
sec_ax.set_xticks([0.75, 2.75], [f"\n{label}" for label in x_ticks_lvl2])
ax.legend(loc="upper right", ncols=1)
ax.set_ylim(0, 1.0)

plt.savefig(Path(__file__).parent / f"plots/{dataset_name}.png", dpi=200)
