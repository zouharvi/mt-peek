#!/usr/bin/env python3

import glob
import matplotlib.pyplot as plt
import jezecek.fig_utils
import numpy as np

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/logs/train_mt_ende_s0_*.log logs/

data = []
for file in glob.glob("logs/train_mt_ende_s0_*.log"):
    if "fully_random" not in file and "ordered_random" not in file:
        continue

    lines = [
        l.rstrip()
        for l in open(file, "r").readlines()
        if "best_bleu " in l
    ]
    if not lines:
        print(file)
        best_bleu = 0
    else:
        best_bleu = float(lines[-1].split("best_bleu ")[-1])
    print(best_bleu)

    if "ordered_random" in file:
        mt_name = "ordered_random"
    else:
        mt_name = "fully_random"

    random_rate = int(file.split("_r")[-1].removesuffix(".log"))
    data.append((mt_name, random_rate, best_bleu))


def plot_bars(data_local, label, offset, style):
    # sort by rate
    data_local.sort(key=lambda x: x[1])
    print([x[2] for x in data_local])
    plt.bar(
        [x + offset for x in range(len(data_local))],
        [x[2] for x in data_local],
        label=label,
        width=0.33,
        linewidth=1.2,
        **style
    )


plt.figure(figsize=(3.5, 2.5))

data_ordered_random = [x for x in data if x[0] == "ordered_random"]
data_fully_random = [x for x in data if x[0] == "fully_random"]

plot_bars(
    data_fully_random,
    "Fully random",
    offset=-0.19,
    style={"color": "white", "hatch": "\\", "edgecolor": "black"}

)

plot_bars(
    data_ordered_random,
    "Ordered random",
    offset=0.19,
    style={"color": "black", "edgecolor": "black"},
)

rates = [x[1] for x in data if x[0] == "ordered_random"]
rates.sort()

plt.xticks(
    [x for x in range(len(rates)) if x % 2 == 0],
    [f"{r}%" for x, r in enumerate(rates) if x % 2 == 0]
)
plt.xlabel("\% of words in reference accessed")
plt.ylabel("BLEU score")

plt.legend()
plt.tight_layout(pad=0.1)
plt.savefig("computed/yapok.pdf")
plt.show()

baseline_bleu = 39.94
for rate, bleu_fully, bleu_ordered in zip(rates, data_fully_random,data_ordered_random):
    fdev = [x.split(" [SEP]")[0] for x in open(f"data/peek/r{rate:0>3}/dev.de", "r")]
    avg_tokens = np.average([x.count(" ")+1 if x else 0 for x in fdev])
    print(
        f"{rate}\\%",
        f"{avg_tokens:.2f}",
        f"{bleu_fully[2]:.2f}",
        f"{(bleu_fully[2]-baseline_bleu)/avg_tokens:.2f}",
        f"{bleu_ordered[2]:.2f}",
        f"{(bleu_ordered[2]-baseline_bleu)/avg_tokens:.2f}",
        sep=" & ",
        end="\\\\\n"
    )