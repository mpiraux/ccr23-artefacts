#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cycler
import seaborn as sns
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
    \usepackage{siunitx}
    """,    
    "pgf.texsystem": "pdflatex",
    "axes.prop_cycle": cycler(color=['#1b9e77','#d95f02','#7570b3'])
})
plt.rcParams.update({
    "legend.fontsize": 8,
    "figure.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

def fmt_ratio(r: float) -> str:
    if r > 1:
        return f"\SI{{{int((r - 1) * 100)}}}{{\%}}"
    else:
        return f"\SI{{{int(((1/r) - 1) * 100)}}}{{\%}}"

cdf_data = results_filtered
sns.ecdfplot(cdf_data, ax=axes[0], x='v4_duration', label='IPv4')
sns.ecdfplot(cdf_data, ax=axes[0], x='v6_duration', label='IPv6')
axes[0].set_xlim(20, 600)
axes[0].set_xlabel("RCT (ms)")
xticks = [20, 200, 400, 600]
axes[0].set_xticks(xticks, list(map(str, xticks)), minor=False)
xmticks = [100, 300, 500]
axes[0].set_xticks(xmticks, [], minor=True)
yticks = [0.0, 0.25, 0.5, 0.75, 1]
axes[0].set_yticks(yticks, list(map(lambda x: f'{x:.2f}', yticks)), minor=False)
axes[0].set_ylabel("CDF")
axes[0].legend(handlelength=1, handletextpad=0.4)
axes[0].grid(color='lightgrey')

ratios_filtered = sorted(results_filtered.v4_v6_ratio)
fig, axes = plt.subplots(1, 2, figsize=(3.34*1.08, 1), dpi=600, gridspec_kw={'width_ratios': [0.3, 0.7], 'wspace': 0.2}, sharey=True)
axes[1].vlines([1], ymin=0, ymax=1, colors='lightgrey', linestyle='--', linewidth=.75)
axes[1].plot(ratios_filtered, [x/len(ratios_filtered) for x in range(len(ratios_filtered))], color='C2')
axes[1].text(1/1.4, 0.66, 'IPv4 faster', fontsize = 9, va='center', ha='center', bbox = dict(facecolor='white', alpha=0.5, boxstyle='larrow'))
axes[1].text(1.4, 0.33, 'IPv6 faster', fontsize = 9, va='center', ha='center', bbox = dict(facecolor='white', alpha=0.5, boxstyle='rarrow'))
axes[1].fill_between(ratios_filtered[:sum(r < 1/1.1 for r in ratios_filtered)], [x/len(ratios_filtered) for x in range(sum(r < 1/1.1 for r in ratios_filtered))], hatch='/////', facecolor='none', edgecolor='gray', linewidth=0.0, rasterized=True)
gc.collect()
axes[1].fill_between(ratios_filtered[-sum(r > 1.1 for r in ratios_filtered):], [x/len(ratios_filtered) for x in range(len(ratios_filtered))][-sum(r > 1.1 for r in ratios_filtered):], hatch='/////', facecolor='none', y2=1, edgecolor='gray', linewidth=0.0, rasterized=True)                                        
axes[1].set_ylim(0, 1)
axes[1].set_xlim(1/2, 2)
axes[1].set_xscale('log')
axes[1].set_xlabel('Ratio of IPv4/IPv6 HTTP RCT')
axes[1].grid(color='lightgrey')
hticks = [1.1, 1.33, 1.60, 2]
ticks = sorted([1/x for x in hticks if x != 1]) + hticks
axes[1].set_xticks(ticks, list(map(fmt_ratio, ticks)))
axes[1].set_xticks([], [], minor=True)
yticks = [0.0, 0.25, 0.5, 0.75, 1]
axes[1].set_yticks(yticks, list(map(lambda x: f'{x:.2f}', yticks)), minor=False)

plt.savefig('ripe_http_ratio_1_8_june_2023.pdf', bbox_inches='tight', pad_inches=0)