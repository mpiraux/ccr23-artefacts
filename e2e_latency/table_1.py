#!/usr/bin/env python3

import pandas as pd

simulation = pd.read_csv('ripe_simulation.csv')
print(len(simulation))
simulation = simulation[simulation.zone.notna()]
print(len(simulation))

# Prints
"""
489589
398374
"""

zones = ['IPv4 is better', 'None is better', 'IPv6 is better']
simulation.zone = simulation.zone.map(lambda x: zones[int(x-1)])
print(simulation[['zone', 'anchor_id']].groupby(by='zone').count())
print(simulation[['zone', 'anchor_id']].groupby(by='zone').count() / len(simulation))

# Prints
"""
                anchor_id
zone                     
IPv4 is better     113092
IPv6 is better     129070
None is better     156212
                anchor_id
zone                     
IPv4 is better   0.283884
IPv6 is better   0.323992
None is better   0.392124
"""