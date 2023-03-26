#!/usr/bin/env python3

import sys
sys.path.append("src")
import glob
import matplotlib.pyplot as plt
import fig_utils

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/logs/train_mt_ende_s0_*.log logs/

data = []
for file in glob.glob("logs/train_mt_ende_s0_*.log"):
    lines = [l.rstrip()
             for l in open(file, "r").readlines() if "best_bleu " in l]
    if not lines:
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
        width=0.3,
        linewidth=2,
        **style
    )

plt.figure(figsize=(3.5, 2.5))

plot_bars(
    [x for x in data if x[0] == "fully_random"],
    "fully random",
    offset=-0.15,
    style={"color": "white", "hatch": "\\", "edgecolor": "black"}

)

plot_bars(
    [x for x in data if x[0] == "ordered_random"],
    "ordered random",
    offset=0.15,
    style={"color": "black", "hatch": "/", "edgecolor": "black"},
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
plt.tight_layout(pad=0.2)
plt.savefig("computed/yapok.pdf")
plt.show()
