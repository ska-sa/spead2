#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv(
        "tut-6-send-pktsize-results.txt",
        sep=" ",
        names=["Language", "Packet size (bytes)", "Rate (MB/s)", "Units"],
    )
    sns.set_theme()
    sns.relplot(
        data=df,
        x="Packet size (bytes)",
        y="Rate (MB/s)",
        hue="Language",
        style="Language",
        aspect=2,
    )
    plt.savefig("tut-6-send-pktsize-plot.svg")


if __name__ == "__main__":
    main()
