#!/usr/bin/env python3
import pandas as pd
import cbor

data = pd.read_csv('data.csv')
del data['v4_v6_ratio']
paths = []
for (prb_id, anchor_id), path_data in data.groupby(by=['prb_id', 'anchor_id'])[['start_time', 'v4_duration', 'v6_duration']]:
        path_data = path_data.sort_values(by='start_time')
        v4_durations, v6_durations = list(path_data.v4_duration.map(float)), list(path_data.v6_duration.map(float))
        paths.append((int(prb_id), int(anchor_id), v4_durations, v6_durations))

with open('paths.cbor') as f:
        data = cbor.load(f)
