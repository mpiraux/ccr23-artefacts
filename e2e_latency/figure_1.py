#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
results = pd.read_csv('data.csv')
results = results.sort_values(by=['prb_id', 'anchor_id', 'start_time'])

lo, hi = 0.05, 0.95
print('v4 5th/95th', results.v4_duration.quantile(lo), results.v4_duration.quantile(hi))
print('v6 5th/95th', results.v6_duration.quantile(lo), results.v6_duration.quantile(hi))
print('Results raw', len(results))
results_filtered = results[(results.v4_duration.quantile(lo) < results.v4_duration) & 
                           (results.v4_duration.quantile(hi) > results.v4_duration) & 
                           (results.v6_duration.quantile(lo) < results.v6_duration) & 
                           (results.v6_duration.quantile(hi) > results.v6_duration)]
print('Results filtered', len(results_filtered))

# Prints
"""
v4 5th/95th 19.7076411 575.4557488
v6 5th/95th 20.1878581 588.5031058000002
Results raw 156826543
Results filtered 137350884
"""

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "text.latex.preamble": r"""
    %\usepackage{amsmath}
    \usepackage[T1]{fontenc}
    \usepackage[tt=false, type1=true]{libertine}
    \usepackage[varqu]{zi4}
    \usepackage[libertine]{newtxmath}
    """,
    "pgf.texsystem": "pdflatex"
})
plt.rcParams.update({
    "legend.fontsize": 8,
    "figure.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

ratios_filtered = sorted(results_filtered.v4_v6_ratio)
plt.figure(figsize=(3.35*1.08, 1.75), dpi=200)
plt.vlines([1], ymin=0, ymax=1, colors='lightgrey', linestyle='--', linewidth=.75)
plt.plot(ratios_filtered, [x/len(ratios_filtered) for x in range(len(ratios_filtered))])
plt.ylim(0, 1)
plt.ylabel('CDF')
plt.xlim(1/2, 2)
plt.xscale('log')
plt.xlabel('Ratio of IPv4/IPv6 HTTP request completion times')
plt.grid(color='lightgrey')
hticks = [1.1, 1.33, 2, 2]
ticks = sorted([1/x for x in hticks if x != 1]) + hticks
plt.xticks(ticks, list(map(lambda x: f'{x:.2f}', ticks)))
plt.xticks([], [], minor=True)
yticks = [0.0, 0.25, 0.5, 0.75, 1]
plt.yticks(yticks, list(map(lambda x: f'{x:.2f}', yticks)), minor=False)
plt.tight_layout()
plt.savefig('ripe_http_ratio_1_8_june_2023.pdf', bbox_inches='tight', pad_inches=0)