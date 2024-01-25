#!/usr/bin/env python3

# ruff: noqa: E501
# To generate the results:
# (for i in {13..26}; do let h=2**i; let n=2**(32-i); echo -n "Python $n $h "; ../examples/tut_8_send_reuse_memory.py -n $n -H $h -p 9000 192.168.31.2 8888; sleep 10; done) > tut-10-send-reuse-heaps-results.txt
# (for i in {13..26}; do let h=2**i; let n=2**(32-i); echo -n "C++ $n $h "; ../build/examples/tut_8_send_reuse_memory -n $n -H $h -p 9000 192.168.31.2 8888; sleep 10; done) >> tut-10-send-reuse-heaps-results.txt

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv(
        "tut-10-send-reuse-heaps-results.txt",
        sep=" ",
        names=["Language", "Heaps", "Heap size (bytes)", "Rate (MB/s)", "Units"],
    )
    sns.set_theme()
    g = sns.relplot(
        data=df,
        x="Heap size (bytes)",
        y="Rate (MB/s)",
        hue="Language",
        style="Language",
        aspect=2,
    )
    g.set(
        xscale="log",
        xticks=df["Heap size (bytes)"],
        xticklabels=[f"$2^{{{round(math.log2(x))}}}$" for x in df["Heap size (bytes)"]],
    )
    plt.savefig("tut-10-send-reuse-heaps-plot.svg")


if __name__ == "__main__":
    main()
