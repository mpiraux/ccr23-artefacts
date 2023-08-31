#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cycler
import os
import json
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "legend.fontsize": 8,
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

dfs = []
for label, fpath in zip(['DSL', 'Cable', '4G FWA', 'Campus'], ['netlogs_dsl.json', 'netlogs_cable.json', 'netlog_4G.json', 'netlogs_campus.json']):
    df = pd.read_json(fpath).set_index(['page', 'domain'])
    df['network'] = label
    dfs.append(df)

netlogs = pd.concat(dfs)
exp3_results = netlogs.reset_index()
exp3_results['bin'] = exp3_results.domain.apply(lambda x: '.'.join(x.split('.')[-2:]))

HEAD=60
TAIL=100
mean_exp3_conn_time = exp3_results.groupby(by=['bin', 'network', 'mode']).apply(lambda x: x.conn_time.tail(-HEAD).mean()).rename('mean_conn_time').reset_index()
counts = exp3_results.groupby(by=['bin', 'network', 'mode']).apply(lambda x: x.conn_time.tail(-HEAD).count()).rename('conn_count')
mean_exp3_conn_time = mean_exp3_conn_time[mean_exp3_conn_time.mean_conn_time.notna()]
mean_exp3_conn_time = mean_exp3_conn_time.join(counts, on=['bin', 'network', 'mode'])
bins = counts.reset_index()[counts.reset_index().conn_count >= TAIL].groupby(by=['bin', 'network']).mode.count()
mean_exp3_conn_time = mean_exp3_conn_time.join(bins, on=['bin', 'network'], rsuffix='_count')
mean_exp3_conn_time = mean_exp3_conn_time[(mean_exp3_conn_time.mode_count == 3) & (mean_exp3_conn_time.conn_count >= TAIL) & mean_exp3_conn_time.mean_conn_time.notna()]

lows = mean_exp3_conn_time[mean_exp3_conn_time['mode'] != 'exp3'].groupby(by=['bin', 'network']).apply(lambda x: x[x.mean_conn_time == x.mean_conn_time.min()])[['mode', 'mean_conn_time']].reset_index().drop(columns='level_2')
highs = mean_exp3_conn_time[mean_exp3_conn_time['mode'] != 'exp3'].groupby(by=['bin', 'network']).apply(lambda x: x[x.mean_conn_time == x.mean_conn_time.max()])[['mode', 'mean_conn_time']].reset_index().drop(columns='level_2')
exp3s = mean_exp3_conn_time[mean_exp3_conn_time['mode'] == 'exp3'].copy()
lows['type'] = 'low'
highs['type'] = 'high'
exp3s['type'] = 'exp3'
data = pd.concat([lows, exp3s, highs]).sort_values(by='mode')

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.35*1.08,2.25), dpi=200, width_ratios=[3/4, 1/4], layout='constrained')
plt.rcParams.update({
    "legend.fontsize": 8,
    "figure.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

ax = sns.violinplot(ax=ax, data=data[data.network != '4G FWA'], x='network', hue='type', y='mean_conn_time', cut=0, order=['Campus', 'DSL', 'Cable'], bw=.15, legend=False, linewidth=1)
ax.set_ylim(ymin=2, ymax=40)
ax.set_xlabel(None)
ax.set_ylabel('Mean handshake time (ms)')

ax2 = sns.violinplot(ax=ax2, data=data[data.network == '4G FWA'], x='network', hue='type', y='mean_conn_time', cut=0, order=['4G FWA'], bw=.15, legend=False, linewidth=1)
ax2.set_ylim(ymin=50, ymax=90)
ax2.legend([],[], frameon=False)
ax2.set_ylabel(None)
ax2.set_xlabel(None)
ax.legend(handles=ax.legend_.legendHandles, labels=['Prototype', 'Best average choice', 'Worst average choice'], loc='upper left')
fig.supxlabel('Access network', x=0.55, y=0.1)
plt.tight_layout()
plt.savefig('browserless.pdf', bbox_inches='tight', pad_inches=0.01)