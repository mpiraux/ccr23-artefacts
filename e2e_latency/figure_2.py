#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cycler

simulation = pd.read_csv('ripe_simulation.csv').drop(columns=['Unnamed: 0'])
simulation = simulation[simulation.zone.notna()]
a = simulation.groupby(by=['prb_id', 'zone'])[['anchor_id']].count().reset_index().rename(columns={'anchor_id': 'total_anchors'})
a['perc_anchors'] = list(a.groupby(by=['prb_id']).apply(lambda x: x.total_anchors/x.total_anchors.sum()))

v4 = []
v6 = []
none = []
v4_n = []
v6_n = []
for prb_id in a.prb_id.unique():
    prb = a[a.prb_id == prb_id]
    zone1 = prb[prb.zone == 1].perc_anchors.max()
    zone3 = prb[prb.zone == 3].perc_anchors.max()
    zone2 = prb[prb.zone == 2].perc_anchors.max()
    largest_zone = max([zone1, zone2])
    if largest_zone != largest_zone:
        continue
    if zone2 == zone2:
        none.append(zone2)
    if zone1 >= zone3:
        v4.append(zone1)
        if zone2 == zone2:
            v4_n.append(zone1 + zone2)
    if zone3 >= zone1:
        v6.append(zone3)
        if zone2 == zone2:
            v6_n.append(zone3 + zone2)

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

fig = plt.figure(figsize=(3.35*1.08*0.61, 1.5), dpi=200)

plt.plot(sorted(v4), [x/len(v4) for x in range(len(v4))], label='IPv4 is better (v4)')
plt.plot(sorted(v6), [x/len(v6) for x in range(len(v6))], label='IPv6 is better (v6)')
plt.plot(sorted(none), [x/len(none) for x in range(len(none))], label='None strongly better (N)')
plt.grid(color='lightgrey')
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.xlim(0, 1)
plt.ylim(0, 1)
lgd = [fig.legend(loc='center right', bbox_to_anchor=(1.60, 0.61))]

plt.xlabel('Ratio of anchors reached')
plt.ylabel('CDF')
plt.tight_layout()
plt.savefig('ripe_http_best_path_1_jun_2023.pdf', bbox_extra_artists=lgd, bbox_inches='tight', pad_inches=0)