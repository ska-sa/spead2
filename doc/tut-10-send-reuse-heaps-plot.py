#!/usr/bin/env python3

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
