#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cycler

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "figure.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.latex.preamble": r"""
    %\usepackage{amsmath}
    \usepackage[T1]{fontenc}
    \usepackage[tt=false, type1=true]{libertine}
    \usepackage[varqu]{zi4}
    \usepackage[libertine]{newtxmath}
    """,
    "pgf.texsystem": "pdflatex",
    "axes.prop_cycle": cycler(color=['#1b9e77','#d95f02','#7570b3'])
})

import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

plt.rcParams.update({
    "legend.fontsize": 9,
})

simulation = pd.read_csv('ripe_simulation.csv')
simulation = simulation[simulation.zone.notna()]
zones = ['IPv4 is better', 'None is strongly better', 'IPv6 is better']
simulation.zone = simulation.zone.map(lambda x: zones[int(x-1)])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.031875, 2), sharey=True, dpi=200)
labels = []
for i, z in enumerate(sorted(simulation.zone.unique())):
    sim_z = simulation[simulation.zone == z]
    label = f'{z} (${len(sim_z.median_best_choice) / len(simulation) * 100:.2f}$ \% sim.)'
    labels.append(label)
    ax1.plot(sorted(sim_z.median_best_choice), [x/len(sim_z) for x in range(len(sim_z))], label=label, color=f'C{i}')
    ax1.plot(sorted(sim_z.static_best_choice), [x/len(sim_z) for x in range(len(sim_z))], linestyle='--', color=f'C{i}')
    bpath_choice = sim_z.best_path_choice.dropna()
    ax1.plot(sorted(bpath_choice), [x/len(bpath_choice) for x in range(len(bpath_choice))], linestyle='-.', color=f'C{i}')

ax1.set_ylabel("CDF")
ax1.set_xlabel("Ratio of lowest \n completion time choices")
ax1.set_yticks(ticks=[0, 0.25, 0.5, 0.75, 1])
ax1.set_ylim(0, 1)
ax1.set_xlim(xmin=simulation.best_path_choice.dropna().min(), xmax=1)
ax1.grid(color='lightgrey')

flows = pd.read_csv('ripe_flows.csv')
flows.best_path = flows.best_path.fillna('none')
flows = flows.drop(columns=['Unnamed: 0'])
flows.best_path = flows.best_path.fillna('none')
gains = simulation.drop(columns=['Unnamed: 0']).set_index(['prb_id', 'anchor_id']).join(flows.set_index(['prb_id', 'anchor_id']), on=['prb_id', 'anchor_id'])

for cid, bp in enumerate(['v4', 'v6', 'none']):
    a = gains[gains.best_path == bp]
    if bp.startswith('None'):
        print(a)
        print(sorted(a.avg_expected_gain * a.median_best_choice))    
    ax2.plot(sorted(a.avg_expected_gain * a.median_best_choice), [x/len(a.avg_expected_gain) for x in range(len(a.
    avg_expected_gain))], color=f'C{cid}')
    ax2.plot(sorted(a.avg_expected_gain), [x/len(a.avg_expected_gain) for x in range(len(a.
    avg_expected_gain))], color=f'C{cid}', linestyle='--')
    v = (a.avg_expected_gain * a.best_path_choice).fillna(-1)
    ax2.plot(sorted(v), [x/len(v) for x in range(len(v))], color=f'C{cid}', linestyle='-.')
   
ax2.set_xlabel('Absolute expected gain \n per request (ms)')
ax2.set_xticks(ticks=[0, 50, 100, 150, 200, 250])
ax2.set_xlim(0, 125)
ax2.set_ylim(0, 1)
ax2.grid(color='lightgrey')

for cid, bp in enumerate(['v4', 'v6', 'none']):
    a = gains[gains.best_path == bp]
    ax3.plot(sorted((a.avg_expected_gain * a.median_best_choice / a.median_duration.fillna(float('inf'))).fillna(-1)), [x/len(a.avg_expected_gain) for x in range(len(a.avg_expected_gain))], color=f'C{cid}')
    ax3.plot(sorted((a.avg_expected_gain / a.median_duration.fillna(float('inf'))).fillna(-1)), [x/len(a.avg_expected_gain) for x in range(len(a.avg_expected_gain))], color=f'C{cid}', linestyle='--')
    v = (a.avg_expected_gain * a.best_path_choice / a.median_duration.fillna(float('inf'))).fillna(-1)
    ax3.plot(sorted(v), [x/len(v) for x in range(len(v))], color=f'C{cid}', linestyle='-.')

ax3.set_xlabel('Expected gain per request \n relative to median RCT')
ax3.set_xlim(0, 0.5)
ax3.set_ylim(0, 1)
ax3.grid(color='lightgrey')
from matplotlib.patches import Patch
custom_lines = [Patch(facecolor='C0'),
                Patch(facecolor='C1'),
                Patch(facecolor='C2'),
                Line2D([0], [0], linestyle='-', color='black', lw=1),
                Line2D([0], [0], linestyle='--', color='black', lw=1),
                Line2D([0], [0], linestyle='-.', color='black', lw=1)]
lgd = fig.legend(flip(custom_lines, 3), flip(labels + ['EXP3', 'Best family for probe \& anchor', 'Best overall family for probe'], 3), loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.175))
plt.tight_layout()
plt.savefig('ripe_simulation.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0)