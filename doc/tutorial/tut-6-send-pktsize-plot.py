#!/usr/bin/env python3

# ruff: noqa: E501
# To collect the data, run this in a shell
# (for p in `seq 1000 100 9000`; do echo -n "Python $p "; ../../examples/tutorial/tut_6_send_pktsize.py 192.168.31.2 8888 -p $p -n 10000; done) > tut-6-send-pktsize-results.txt
# (for p in `seq 1000 100 9000`; do echo -n "C++ $p "; ../../build/examples/tutorial/tut_6_send_pktsize 192.168.31.2 8888 -p $p -n 10000; done) >> tut-6-send-pktsize-results.txt

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
