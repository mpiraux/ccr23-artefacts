#!/usr/bin/env python3

import multiprocessing
import random
import pandas as pd
import numpy as np
import cbor

from scipy import stats
from tqdm import tqdm
import ruptures as rpt

from exp3 import EXP3

def get_break_points(signal):
    signal = np.array(signal)
    model = "l2"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=30, jump=5).fit(signal)
    return algo.predict(pen=5)

def get_zone(v4_durations, v6_durations):
    v4_durations_f = sorted(v4_durations)[int(len(v4_durations) * 0.1):int(len(v4_durations) * 0.9)]
    v6_durations_f = sorted(v6_durations)[int(len(v6_durations) * 0.1):int(len(v6_durations) * 0.9)]
    result = stats.ttest_ind(v4_durations_f, v6_durations_f, equal_var=False)
    low, high = result.confidence_interval()
    if result.pvalue > 0.02 or (low > 0 and min(map(abs, [low, high])) < np.std(v4_durations_f)) or (low < 0 and min(map(abs, [low, high])) < np.std(v6_durations_f)):
        zone = 2
    else:
        if low > 0:
            zone = 3
        else:
            zone = 1
    return zone

def change_points(v):
    prb_id, anchor_id, v4_durations, v6_durations = v
    if len(v4_durations + v6_durations) < 300:
        return prb_id, anchor_id, [], False

    break_points = sorted(set(sum(map(get_break_points, [v4_durations, v6_durations]), [])))
    break_points = [0] + break_points
    zones = []
    segments = []
    for s, e in zip(break_points[:-1], break_points[1:]):
        z = get_zone(v4_durations[s:e], v6_durations[s:e])
        zones.append(z)
        segments.append((s, e, z))
    return prb_id, anchor_id, segments, 1 in zones and 3 in zones


def best_path(v):
    prb_id, anchor_id, v4_durations, v6_durations = v
    if len(v4_durations + v6_durations) < 300:
        return prb_id, anchor_id, '', 0.0, 0.0
    v4_durations_f = sorted(v4_durations)[int(len(v4_durations) * 0.1):int(len(v4_durations) * 0.9)]
    v6_durations_f = sorted(v6_durations)[int(len(v6_durations) * 0.1):int(len(v6_durations) * 0.9)]
    median_v4 = v4_durations_f[len(v4_durations_f) // 2]
    median_v6 = v6_durations_f[len(v6_durations_f) // 2]
    result = stats.ttest_ind(v4_durations_f, v6_durations_f, equal_var=False)
    low, high = result.confidence_interval()
    if result.pvalue > 0.02:
        return prb_id, anchor_id, '', 0.0, 0.0
    if low > 0 and min(map(abs, [low, high])) < np.std(v4_durations_f):
        return prb_id, anchor_id, '', abs(high + low) / 2, median_v4
    if low < 0 and min(map(abs, [low, high])) < np.std(v6_durations_f):
        return prb_id, anchor_id, '', abs(high + low) / 2, median_v6
    if low > 0:
        return prb_id, anchor_id, 'v6', abs(high + low) / 2, median_v4
    else:
        return prb_id, anchor_id, 'v4', abs(high + low) / 2, median_v6

def simulate_path(v):
    prb_id, anchor_id, v4_durations, v6_durations, prb_best_path = v
    if len(v4_durations + v6_durations) < 300:
        return prb_id, anchor_id, float('nan'), float('nan')
    choices = {}
    gains = {}
    TIME_TO_CONVERGE = 30
    for gamma in [0.1]:
        c = []
        g = []
        r = random.Random("test")
        for _ in range(100):
            exp3 = EXP3(2, gamma=gamma, r=r, max_weight=100.0)
            last_v4, last_v6 = None, None
            for _ in range(1):
                best_choice_count = 0
                gain = 0
                for v4_d, v6_d in zip(v4_durations, v6_durations):
                    action = exp3.take_action()
                    reward = 0.0
                    if action == 0:
                        if last_v6 is not None and v4_d <= last_v6:
                            reward = 1.0
                        if v4_d <= v6_d and exp3.round > TIME_TO_CONVERGE:
                            best_choice_count += 1
                        last_v4 = v4_d
                        gain += v6_d - v4_d
                    else:
                        if last_v4 is not None and v6_d <= last_v4:
                            reward = 1.0
                        if v6_d <= v4_d and exp3.round > TIME_TO_CONVERGE:
                            best_choice_count += 1
                        last_v6 = v6_d
                        gain += v4_d - v6_d
                    exp3.give_reward(action, reward)
            c.append(best_choice_count / (len(v4_durations) - TIME_TO_CONVERGE))
            g.append(gain)
        choices[gamma] = c
        gains[gamma] = g
    
    *_, bpath, avg_distance, median_duration = best_path(v[:4])
    if bpath == 'v4':
        zone = 1
    elif bpath == '':
        zone = 2
    else:
        zone = 3
        

    median_g0_1_bchoice = sorted(choices[0.1])[len(choices[0.1])//2]
    if prb_best_path == 'v4':
        best_path_choice = sum(v4 <= v6 for v4, v6 in zip(v4_durations, v6_durations)) / len(v4_durations)
    elif prb_best_path == 'v6':
        best_path_choice = sum(v6 <= v4 for v4, v6 in zip(v4_durations, v6_durations)) / len(v4_durations)
    else:
        best_path_choice = float('nan')
    static_best_choice = max((sum(v4 <= v6 for v4, v6 in zip(v4_durations, v6_durations)) / len(v4_durations), sum(v6 <= v4 for v4, v6 in zip(v4_durations, v6_durations)) / len(v4_durations)))
    
    return prb_id, anchor_id, zone, avg_distance, median_duration, median_g0_1_bchoice, best_path_choice, static_best_choice

with open('paths.cbor', 'rb') as f:
    http_data = cbor.load(f)

with multiprocessing.Pool() as p:
    results = list(tqdm(p.imap_unordered(change_points, http_data), total=len(http_data)))
    pd.DataFrame(sorted(results), columns=['prb_id', 'anchor_id', 'segments', 'mixed_zones']).to_csv('ripe_change_points.csv')


with multiprocessing.Pool() as p:
    results = list(tqdm(p.imap_unordered(best_path, http_data), total=len(http_data)))
    probes = {}
    for prb_id, anchor_id, bpath, avg_expected_gain, median_duration in sorted(results):
        l = probes.get(prb_id, [])
        l.append(bpath)
        probes[prb_id] = l

    pd.DataFrame(sorted(results), columns=('prb_id', 'anchor_id', 'best_path', 'avg_expected_gain', 'median_duration')).to_csv('ripe_flows.csv')
    for i in range(len(http_data)):
        prb_id, *_ = http_data[i]
        best_paths = dict(np.array(np.unique(probes[prb_id], return_counts=True)).T)
        if int(best_paths.get('v4', 0)) > int(best_paths.get('v6', 0)) and int(best_paths.get('v4', 0)) > int(best_paths.get('', 0)):
            bp = 'v4'
        elif int(best_paths.get('v6', 0)) > int(best_paths.get('v4', 0)) and int(best_paths.get('v6', 0)) > int(best_paths.get('', 0)):
            bp = 'v6'
        else:
            bp = None
        http_data[i] += [bp]


with multiprocessing.Pool() as p:
    results = list(tqdm(p.imap_unordered(simulate_path, http_data), total=len(http_data)))
    pd.DataFrame(results, columns=('prb_id', 'anchor_id', 'zone', 'avg_expected_gain', 'median_duration','median_best_choice', 'best_path_choice', 'static_best_choice')).to_csv('ripe_simulation.csv')