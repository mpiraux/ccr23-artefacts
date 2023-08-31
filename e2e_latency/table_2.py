#!/usr/bin/env python3

import json
import pandas as pd

change_points = pd.read_csv('ripe_change_points.csv', converters={'segments': lambda x: json.loads(x.replace('(', '[').replace(')', ']')) or None}).drop(columns=['Unnamed: 0'])
change_points = change_points[change_points.segments.notnull()]
change_points = change_points.explode('segments')
change_points = pd.concat([change_points, change_points.segments.apply(pd.Series)], axis=1).drop(columns=['segments'])

simulation = pd.read_csv('ripe_simulation.csv').drop(columns=['Unnamed: 0'])
simulation = simulation[simulation.zone.notna()]
segments = change_points.set_index(['prb_id', 'anchor_id']).join(simulation.set_index(['prb_id', 'anchor_id']))

cat_1 = len(segments[((segments.zone == 1) & (segments.segment_zone == 3)) | ((segments.zone == 3) & (segments.segment_zone == 1))][['prb_id', 'anchor_id']].drop_duplicates())
cat_2 = len(segments[((segments.zone == 1) & (segments.segment_zone == 2)) | ((segments.zone == 3) & (segments.segment_zone == 2))][['prb_id', 'anchor_id']].drop_duplicates())
cat_3 = len(segments[(segments.zone == 2) & (segments.segment_zone.isin([1, 3]))][['prb_id', 'anchor_id']].drop_duplicates())
total = len(pd.read_csv('ripe_simulation.csv').zone.dropna()) 
cat_4 = total - cat_1 - cat_2 - cat_3

print(f'{cat_1} ({cat_1 / total * 100:.2f} \%) & {cat_2} ({cat_2 / total * 100:.2f} \%) & {cat_3} ({cat_3 / total * 100:.2f} \%) & {cat_4} ({cat_4 / total * 100:.2f} \%)')

# Prints

"""
4569 (1.15 \%) & 32459 (8.15 \%) & 27312 (6.86 \%) & 334034 (83.85 \%)
"""