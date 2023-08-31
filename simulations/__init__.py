from typing import Dict, List, Tuple, Any
import random
import hashlib
from math import dist, log2

from exp3 import EXP3
from network import Network, TimeseriesNetwork, Flow, NewRenoFlow

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def _isnan(x: float) -> bool:
    return x != x

class Internet:
    destinations: Dict[str, Network]

    def __init__(self, destinations: Dict[str, Network]) -> None:
        self.destinations = destinations
    
    def run_flow(self, destination: str, flow_packets: int, path_idx=None) -> Flow:
        assert destination in self.destinations
        if path_idx is None:
            path_idx = random.choice(range(self.destinations[destination].no_paths))
        flow = NewRenoFlow(path_idx, packets=flow_packets)
        while not flow.completed:
            flow.advance(self.destinations[destination])
        return flow

class StaticInternet(Internet):
    path_idx: int = 0

    def __init__(self, destinations: Dict[str, Network], path_idx: int) -> None:
        super().__init__(destinations)
        self.path_idx = path_idx

    def run_flow(self, destination: str, flow_packets: int, path_idx=None) -> Flow:
        return super().run_flow(destination, flow_packets, path_idx=self.path_idx)

class PingInternet(Internet):
    ping_history: Dict[str, Dict[str, int]] = {}
    draw: int = 1
    lifetime: int = 10
    rewind_prng: bool = True

    def __init__(self, destinations: Dict[str, Network], lifetime=10, draw=1, rewind_prng=True) -> None:
        super().__init__(destinations)
        self.lifetime = lifetime
        self.rewind_prng = rewind_prng
        self.draw = draw
        for d in destinations:
            self.ping_history[d] = {'lifetime': 0, 'best_path_idx': None}

    def run_flow(self, destination: str, flow_packets: int, path_idx=None) -> Flow:
        if destination not in self.ping_history or self.ping_history[destination]['lifetime'] == 0:
            lower_ping = float('inf')
            lower_ping_path_idx = 0
            for path_idx in range(self.destinations[destination].no_paths):
                net = self.destinations[destination]
                if self.rewind_prng:
                    rstate = net.get_state()
                ping = sum([net.draw_rtt(path_idx) for _ in range(self.draw)]) / self.draw
                if ping < lower_ping:
                    lower_ping_path_idx = path_idx
                    lower_ping = ping
                if self.rewind_prng:
                    net.set_state(rstate)
            self.ping_history[destination]['lifetime'] = self.lifetime
            self.ping_history[destination]['best_path_idx'] = lower_ping_path_idx
            
        self.ping_history[destination]['lifetime'] -= 1
        return super().run_flow(destination, flow_packets, path_idx=self.ping_history[destination]['best_path_idx'])

class MeanHistoryInternet(Internet): # TODO Make this share code with EXP3 and KMeans
    flow_history: Dict[str, Dict[int, Any]]
    _history_len = 10

    def __init__(self, destinations: Dict[str, Network], history_len=10) -> None:
        super().__init__(destinations)
        self.flow_history = {d: {path_idx: [] for path_idx in range(self.destinations[d].no_paths)} for d in self.destinations}
        self._history_len = history_len

    def run_flow(self, destination: str, flow_packets: int) -> Flow:
        lowest_mean_path_idx = None
        mean_rtts = [self.mean_rtt(destination, path_idx) for path_idx in range(self.destinations[destination].no_paths)]
        if all(not _isnan(rtt) for rtt in mean_rtts):
            lowest_mean_path_idx = sorted(range(self.destinations[destination].no_paths), key=lambda path_idx: mean_rtts[path_idx])[0]
        flow = super().run_flow(destination, flow_packets, lowest_mean_path_idx)
        history = self.flow_history[destination][flow.path_idx]
        if type(history) == list:
            history.append(flow.smooth_rtt)
            if len(history) == self._history_len:
                history = sum(history) / len(history)
        else:
            history = (history * ((self._history_len - 1) / self._history_len)) + (flow.smooth_rtt * (1 / self._history_len))
        self.flow_history[destination][flow.path_idx] = history
        return flow

    def mean_rtt(self, destination: str, path_idx: int) -> float:
        flow_history = self.flow_history[destination][path_idx]
        if not flow_history:
            return float('nan')
        if type(flow_history) != list:
            return flow_history
        #return float('nan')
        return sum(f for f in flow_history) / len(flow_history)

class OracleInternet(Internet):
    def run_flow(self, destination: str, flow_packets: int, path_idx=None) -> Flow:
        lowest_fct = float('inf')
        lowest_fct_path_idx = 0
        for path_idx in range(self.destinations[destination].no_paths):
            rstate = self.destinations[destination].get_state()
            flow = super().run_flow(destination, flow_packets, path_idx)
            if flow.completion_time < lowest_fct:
                lowest_fct_path_idx = path_idx
                lowest_fct = flow.completion_time
            self.destinations[destination].set_state(rstate)
        return super().run_flow(destination, flow_packets, lowest_fct_path_idx)

class HappyEyeballsInternet(Internet):
    def __init__(self, destinations: Dict[str, Network], connection_attempt_delay: int = 10) -> None:
        super().__init__(destinations)
        self.connection_attempt_delay = connection_attempt_delay

    def run_flow(self, destination: str, flow_packets: int, path_idx=None) -> Flow:
        assert destination in self.destinations
        assert self.destinations[destination].no_paths == 2, ("HE can't work with more than two paths: ", self.destinations[destination].no_paths)
        rstate = self.destinations[destination].get_state()
        v4_rtt = self.destinations[destination].draw_rtt(0)
        v6_rtt = self.destinations[destination].draw_rtt(1)
        path_idx = 1
        if v4_rtt + self.connection_attempt_delay < v6_rtt:
            path_idx = 0
        # TODO account for delay increase in flow completion time
        self.destinations[destination].set_state(rstate)
        self.destinations[destination].draw_rtt(path_idx)  # Steps random state               
        return super().run_flow(destination, flow_packets, path_idx)
    

class EXP3Internet(Internet):
    exp3: List[EXP3]
    flow_history: Dict[str, Dict[str, Any]]

    def __init__(self, destinations: Dict[str, Network], exp3_instances: int = 10, gamma=0.1, exp3_max_weight: float = 10000.0) -> None:
        super().__init__(destinations)
        self.flow_history = {d: {'max': None} for d in self.destinations}
        no_paths = [destinations[d].no_paths for d in destinations]
        assert all([p == no_paths[0] for p in no_paths])
        self.exp3 = [EXP3(no_paths[0], gamma=gamma, max_weight=exp3_max_weight, enable_history=False) for _ in range(exp3_instances)]

    def get_exp3(self, destination: str) -> EXP3:
        def stable_hash(s: str) -> int:
            # TODO: This should vary from run to run
            return int((hashlib.sha512(s.encode())).hexdigest(), 16)
        if len(self.destinations) == len(self.exp3):
            return self.exp3[sorted(list(self.destinations)).index(destination)]
        return self.exp3[stable_hash(destination) % len(self.exp3)]

    def get_reward(self, destination: str, flow: Flow) -> float:
        destination_max_rtt = self.flow_history[destination]['max']
        if destination_max_rtt is None:
            destination_max_rtt = flow.smooth_rtt
        destination_max_rtt = max(destination_max_rtt, flow.smooth_rtt)
        raw_reward = (destination_max_rtt - flow.smooth_rtt) / destination_max_rtt
        return raw_reward

    def run_flow(self, destination: str, flow_packets: int) -> Flow:
        assert destination in self.destinations
        exp3 = self.get_exp3(destination)
        path_idx = exp3.take_action()
        flow = super().run_flow(destination, flow_packets, path_idx)
        exp3.give_reward(path_idx, self.get_reward(destination, flow))
        self.flow_history[destination]['max'] = max(self.flow_history[destination]['max'] or 0, flow.smooth_rtt)
        return flow

class EXP3BinaryReward(EXP3Internet):
    def get_reward(self, destination: str, flow: Flow) -> float:
        history = self.flow_history[destination]
        reward = float(all(flow.smooth_rtt <= history.get(path_idx, float('inf')) for path_idx in range(self.destinations[destination].no_paths) if path_idx != flow.path_idx))
        self.flow_history[destination][flow.path_idx] = flow.smooth_rtt
        return reward

class EXP3BinaryRewardThroughput(EXP3Internet):
    def get_reward(self, destination: str, flow: Flow) -> float:
        history = self.flow_history[destination]
        flow_throughput = log2(flow.no_packets) / flow.completion_time / 1000
        reward = float(all(flow_throughput >= history.get(path_idx, 0) for path_idx in range(self.destinations[destination].no_paths) if path_idx != flow.path_idx))
        self.flow_history[destination][flow.path_idx] = flow_throughput
        return reward


class KMeansExp3Internet(EXP3Internet):
    exp3_centers: List[Tuple[float, float]]
    _flow_history_len: int = 20
    _kmeans = None
    _kmeans_validity: int = 250

    def __init__(self, destinations: Dict[str, Network], exp3_instances: int = 10, gamma=0.1, exp3_max_weight: float = 10000.0, kmeans_n_init=3, kmeans_validity=1000, history_len=100) -> None:
        super().__init__(destinations, exp3_instances, gamma, exp3_max_weight)
        no_paths = next(iter(self.destinations.values())).no_paths 
        assert all(self.destinations[d].no_paths == no_paths for d in destinations), 'All destinations must have the same number of paths'
        for d in self.destinations:
            for path_idx in range(self.destinations[d].no_paths):
                self.flow_history[d][path_idx] = []
        self._kmeans_n_init = kmeans_n_init
        self._kmeans_validity = kmeans_validity
        self._kmeans_expiry = 0
        self.exp3_centers = [(0.0,) * no_paths for _ in range(exp3_instances)]
        self._flow_history_len = history_len

    def get_exp3(self, destination: str) -> EXP3:
        mean_rtts = {}
        for d in self.destinations:
            dest_rtts = [self.mean_rtt(d, path_idx) for path_idx in range(self.destinations[d].no_paths)]
            if all(not _isnan(rtt) for rtt in dest_rtts):
                mean_rtts[d] = dest_rtts
        if destination in mean_rtts and len(mean_rtts) >= len(self.exp3):
            if self._kmeans is None or self._kmeans_expiry == 0:
                self._kmeans = KMeans(n_clusters=len(self.exp3), random_state=0, n_init=self._kmeans_n_init).fit(list(mean_rtts.values()))
                self._kmeans_expiry = self._kmeans_validity
                assigned_centers = set()
                for cluster_coords in self._kmeans.cluster_centers_:
                    closest_exp3_idx, _ = sorted([(idx, exp3) for idx, exp3 in enumerate(self.exp3_centers) if idx not in assigned_centers], key=lambda x: dist(cluster_coords, x[1]))[0]
                    self.exp3_centers[closest_exp3_idx] = cluster_coords
                    assert closest_exp3_idx not in assigned_centers
                    assigned_centers.add(closest_exp3_idx)
            centroid = self._kmeans.predict([mean_rtts[destination]])[0]
            cluster_id, _ = sorted(enumerate(self.exp3_centers), key=lambda x: dist(self._kmeans.cluster_centers_[centroid], x[1]))[0]
            self._kmeans_expiry -= 1
            return self.exp3[cluster_id]
        else:
            return EXP3(self.destinations[d].no_paths, gamma=1, enable_history=False)
            #return super().get_exp3(destination)

    def run_flow(self, destination: str, flow_packets: int) -> Flow:
        flow = super().run_flow(destination, flow_packets)
        history = self.flow_history[destination][flow.path_idx]
        if type(history) == list:
            history.append(flow.smooth_rtt)
            if len(history) == self._flow_history_len:
                history = sum(history) / len(history)
        else:
            history = (history * (self._flow_history_len - 1) / self._flow_history_len) + (flow.smooth_rtt * (1 / self._flow_history_len))
        self.flow_history[destination][flow.path_idx] = history
        return flow

    def mean_rtt(self, destination: str, path_idx: int) -> float:
        flow_history = self.flow_history[destination][path_idx]
        if not flow_history:
            return float('nan')
        if type(flow_history) != list:
            return flow_history
        return sum(f for f in flow_history) / len(flow_history)

class KMeansRewardEXP3Internet(KMeansExp3Internet):
    def get_kmeans_reward(self, destination: str, flow_srtt: float) -> float:
        destination_max_rtt = self.flow_history[destination]['max']
        if destination_max_rtt is None:
            destination_max_rtt = flow_srtt
        destination_max_rtt = max(destination_max_rtt, flow_srtt)
        raw_reward = (destination_max_rtt - flow_srtt) / destination_max_rtt
        return raw_reward

    def get_exp3(self, destination: str) -> EXP3:
        features = {}
        for d in self.destinations:
            dest_rtts = [self.mean_rtt(d, path_idx) for path_idx in range(self.destinations[d].no_paths)]
            if all(not _isnan(rtt) for rtt in dest_rtts):
                features[d] = [self.get_kmeans_reward(d, mrtt) for mrtt in dest_rtts]
        if destination in features and len(features) >= len(self.exp3):
            if self._kmeans is None or self._kmeans_expiry == 0:
                self._kmeans = KMeans(n_clusters=len(self.exp3), random_state=0, n_init=self._kmeans_n_init).fit(list(features.values()))
                self._kmeans_expiry = self._kmeans_validity
                assigned_centers = set()
                for cluster_coords in self._kmeans.cluster_centers_:
                    closest_exp3_idx, _ = sorted([(idx, exp3) for idx, exp3 in enumerate(self.exp3_centers) if idx not in assigned_centers], key=lambda x: dist(cluster_coords, x[1]))[0]
                    self.exp3_centers[closest_exp3_idx] = cluster_coords
                    assert closest_exp3_idx not in assigned_centers
                    assigned_centers.add(closest_exp3_idx)
            centroid = self._kmeans.predict([features[destination]])[0]
            cluster_id, _ = sorted(enumerate(self.exp3_centers), key=lambda x: dist(self._kmeans.cluster_centers_[centroid], x[1]))[0]
            self._kmeans_expiry -= 1
            return self.exp3[cluster_id]
        else:
            return super().get_exp3(destination)

class KMeansEXP3BinaryRewardInternet(EXP3BinaryReward, KMeansExp3Internet):
    def __init__(self, destinations: Dict[str, Network], exp3_instances: int = 10, gamma=0.1, exp3_max_weight: float = 10000) -> None:
        super().__init__(destinations, exp3_instances, gamma, exp3_max_weight)
        self._flow_history_len = 20
        self._kmeans_validity = 250

    def get_reward(self, destination: str, flow: Flow) -> float:
        history = self.flow_history[destination]
        reward = float(all(flow.smooth_rtt <= history.get(f'last_path_{path_idx}', float('inf')) for path_idx in range(self.destinations[destination].no_paths) if path_idx != flow.path_idx))
        self.flow_history[destination][f'last_path_{flow.path_idx}'] = flow.smooth_rtt
        return reward


if __name__ == "__main__":
    def create_networks_from_pings(*data_dirs: str, expected_number_of_networks) -> Dict[str, Network]:
        import os
        import pandas as pd
        destinations: Dict[str, Network] = {}
        for e in sorted(os.listdir(data_dirs[0])):
            all_pings = []
            if not e.endswith('.csv'):
                    continue
            if not all(os.path.exists(os.path.join(d, e)) for d in data_dirs):
                print(e, "not in all data dirs")
                continue
            for dir in data_dirs:
                pings = pd.read_csv(os.path.join(dir, e))
                for dest, c in pings.probe_dst_addr.value_counts().items():
                    if c < 500:
                        pings = pings[pings.probe_dst_addr != dest]
                for pp in [1, 58]:
                    pp_pings = list(pings[pings['probe_protocol'] == pp].rtt / 10)
                    if not pp_pings or len(pp_pings) < 500:
                        continue
                    all_pings.append(pp_pings)
            if len(all_pings) != expected_number_of_networks:
                print(dir, e, "skipping")
                continue
            destination = os.path.splitext(e)[0]
            destinations[destination] = Network(all_pings)
            #destinations[destination].shifts = [10.0, 0.0, 0.0]
        return destinations
    
    def create_timeseries_networks_from_sync_monitors_pings(*pings_and_names, expected_number_of_networks) -> Dict[str, TimeseriesNetwork]:
        import os
        import pandas as pd
        destinations: Dict[str, TimeseriesNetwork] = {}
        path_idx = 0
        dfs = []
        min_max_ts = float('inf')
        for (ping_filename, names_filename) in pings_and_names:
            resolved = pd.read_csv(names_filename, index_col='qname')
            address_to_name = resolved.melt(ignore_index=False, value_vars=['v4', 'v6'], var_name='version', value_name='address').reset_index().rename(columns={'address': 'probe_dst_addr'}).set_index('probe_dst_addr')
            pings = pd.read_csv(ping_filename, usecols=['capture_timestamp', 'probe_protocol', 'probe_dst_addr', 'rtt'])
            pings.probe_dst_addr = pings.probe_dst_addr.apply(lambda s: s.replace('::ffff:', ''))
            pings = pings.join(address_to_name, on='probe_dst_addr')
            pings.rtt /= 10
            pings['path_idx'] = -1
            for idx, pp in enumerate(pings.probe_protocol.unique()):
                pp_pings = list(pings[pings.probe_protocol == pp].rtt)
                if len(pp_pings) < 10000:
                    continue
                pings.loc[pings.probe_protocol == pp,'path_idx'] = path_idx
                path_idx += 1
            pings = pings[pings.path_idx != -1]
            if pings.capture_timestamp.max() < min_max_ts:
                min_max_ts = pings.capture_timestamp.max()
            dfs.append(pings)
        pings = pd.concat(dfs)
        pings = pings[pings.capture_timestamp <= min_max_ts]
        del dfs
        for name in list(pings.qname.unique()):
            qname_pings = pings[pings.qname == name].sort_values(by='capture_timestamp')
            if len(qname_pings.path_idx.unique()) != expected_number_of_networks:
                print("Skipping", name, qname_pings.path_idx.unique())
                pings = pings[pings.qname != name]
                continue
            all_pings = [list(qname_pings[(qname_pings.path_idx == path_idx)].rtt) for path_idx in range(expected_number_of_networks)]
            all_timestamps = [list(qname_pings[(qname_pings.path_idx == path_idx)].capture_timestamp) for path_idx in range(expected_number_of_networks)]
            destinations[name] = TimeseriesNetwork(all_pings, all_timestamps)
        print("Loaded pings from", pings.capture_timestamp.min(), "to", pings.capture_timestamp.max(), f"({pings.capture_timestamp.max() - pings.capture_timestamp.min()} sec)", "towards", len(destinations), "destinations over", expected_number_of_networks, "paths")
        del pings
        return destinations
       
    def test_internet():
        import pandas as pd
        import matplotlib.pyplot as plt    
        import tqdm
        random.seed("test")
        destinations = create_timeseries_networks_from_sync_monitors_pings(
            #('../caracal_data/monitor/sync/uclm_tadaam_sync.csv', '../caracal_data/monitor/sync/uclr_tadaam_sync.csv'),
            #('../caracal_data/monitor/sync/uclm_starlink_sync.csv', '../caracal_data/monitor/sync/uclr_starlink_sync.csv'),
            ('../caracal_data/monitor/sync/uclm_proximus_sync.csv', '../caracal_data/monitor/sync/uclr_proximus_sync.csv'),
            ('../caracal_data/monitor/sync/uclm_voo_sync.csv', '../caracal_data/monitor/sync/uclr_voo_sync.csv'),
        expected_number_of_networks=4)
        #for k, v in destinations.items():
        #    if k == 'edge.microsoft.com':
        #        destinations = {k: v}
        #        break
        destination_states = [destinations[d].get_state() for d in destinations]
        
        FLOWS = 500 * len(destinations)
        flows_duration_range = (4, 100)

        avg_srtt = {}
        raw_srtt = {}
        runs = 10
        for seed in range(runs):
            for s, d in zip(destination_states, destinations):
                destinations[d].set_state(s)
            random.seed(f"test{'' if seed == 0 else str(seed)}")
            flows_specs = [(random.choice(list(destinations)), random.randrange(*flows_duration_range)) for _ in range(FLOWS)]
            experiments = [
                #(Internet(destinations), 'All random'),
                (StaticInternet(destinations, path_idx=0), 'Always Path 0'),
                #(StaticInternet(destinations, path_idx=1), 'Always Path 1'),
                #(StaticInternet(destinations, path_idx=2), 'Always Path 2'),
                #(StaticInternet(destinations, path_idx=3), 'Always Path 3'),
                #(StaticInternet(destinations, path_idx=1), 'Always Starlink v4'),
                (PingInternet(destinations, draw=4), 'Best ping 10f 4x'),
                (MeanHistoryInternet(destinations), 'Mean history 10f'),
                #(HappyEyeballsInternet(destinations, connection_attempt_delay=10), 'Happy Eyeballs 10ms'),
                #(HappyEyeballsInternet(destinations, connection_attempt_delay=100), 'Happy Eyeballs 100ms'),
                #(EXP3Internet(destinations), '10C Random clusters'),
                #(EXP3Internet(destinations, exp3_instances=len(destinations)), '1C per destination'),
                #(KMeansExp3Internet(destinations), '10C KMeans'),
                #(KMeansExp3Internet(destinations, gamma=0.05), '10C KMeans gamma=0.05'),
                #(KMeansRewardEXP3Internet(destinations), '10C KMeans Reward'),
                (EXP3BinaryReward(destinations, exp3_instances=len(destinations)), "1C/dest EXP3 BR"),
                (EXP3BinaryReward(destinations, exp3_instances=len(destinations), exp3_max_weight=100, gamma=0.2), "1C/dest EXP3 BR Opt"),
                #(EXP3BinaryRewardThroughput(destinations, exp3_instances=len(destinations)), "1C/dest EXP3 BR Throughput"),
                (KMeansEXP3BinaryRewardInternet(destinations), '10C KMeans BR'),
                (OracleInternet(destinations), 'Oracle'),
            ]  # [(Internet, label)]
            for inet, label in experiments:
                for s, d in zip(destination_states, destinations):
                    destinations[d].set_state(s)
                flows = []
                for fs in tqdm.tqdm(flows_specs):
                    if False and len(flows) == len(flows_specs) / 2:
                        impaired_destinations = random.choices(list(inet.destinations), k=len(inet.destinations) // 5)
                        for d in impaired_destinations:
                            d = inet.destinations[d]
                            lowest_rtt_family = d.lowest_rtt_family
                            other_family = 6 if lowest_rtt_family == 4 else 4
                            d.shifts[other_family] = (d.mean_rtt(lowest_rtt_family) - d.mean_rtt(other_family)) - (d.mean_rtt(lowest_rtt_family) / 2)
                    flows.append(inet.run_flow(*fs))
                l = avg_srtt.get(label, [])
                l.extend(pd.Series([flow.completion_time for flow in flows]).ewm(alpha=0.001, min_periods=1).mean())
                avg_srtt[label] = l
                l = raw_srtt.get(label, [])
                l.extend([flow.completion_time for flow in flows])
                raw_srtt[label] = l

        samples = min(50000, len(flows_specs) * runs)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        ax.set_prop_cycle(color=plt.get_cmap('tab20').colors)
        for label, srtt in avg_srtt.items():
            ax.plot(sorted(random.sample(srtt, samples)), [i / samples for i in range(samples)], label=label)
        ax.set_ylabel('CDF')
        ax.set_ylim(0, 1)
        #ax.set_xlabel('sRTT of completed flows')
        ax.set_xlabel('Completion time of flows')
        #ax.set_xlim(15, 30)
        #ax.set_xlim(80, 200)
        ax.legend()
        plt.savefig('last_fig.pdf')
        plt.show()
        return

        inet = experiments[-2][0]
        fig, axes = plt.subplots(len(inet.exp3), 1, sharex=True)
        for idx, ex in enumerate(inet.exp3):
            axes[idx].plot(list(ex._history.keys()), [ex._history[k]['probabilities'][0] for k in ex._history], label='P(v4)')
            axes[idx].plot(list(ex._history.keys()), [ex._history[k]['probabilities'][1] for k in ex._history], label='P(v6)')
            #axes[idx].step(list(ex._history.keys()), [ex._history[k]['reward'] for k in ex._history], label='P(v4)')
            #axes[idx].plot(list(ex._history.keys()), [ex._history[k]['action'] for k in ex._history], label='P(v6)')
            axes[idx].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()
        return

        fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        for label, srtt in avg_srtt.items():
            ax.plot(srtt, label=label)
        ax.set_ylabel('sRTT of the completed flow')
        ax.set_ylim(15, 30)
        ax.set_title("Flows RTT over time")
        ax.legend()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        for label, srtt in avg_srtt.items():
            baseline = list(avg_srtt.values())[0]
            ax.plot([baseline[idx] - v for idx, v in enumerate(srtt)], label=label)
        ax.set_ylabel('sRTT gain (ms)')
        ax.set_ylim(-5, 10)
        ax.set_title("sRTT gain using several techniques compared to a random v4/v6 choice")
        ax.legend()
        plt.show()

    test_internet()
    def test_single_destination():
        import pandas as pd
        import matplotlib.pyplot as plt
        import random
        random.seed("test")
        #pings = pd.read_csv('../caracal_data/office.com@home.csv')
        #v4_pings, v4_ts = list(pings[pings['probe_protocol'] == 1].rtt / 10), list(pings[pings['probe_protocol'] == 1].capture_timestamp)
        #v6_pings, v6_ts = list(pings[pings['probe_protocol'] == 58].rtt / 10), list(pings[pings['probe_protocol'] == 58].capture_timestamp)
        #net = Network([v4_pings, v6_pings])
        pings = pd.read_csv('../edge.microsoft.com_prox_voo_v4.csv').sort_values(by='capture_timestamp')
        p0_pings, p0_ts = list(pings[pings.provider == 'Proximus'].rtt), list(pings[pings.provider == 'Proximus'].capture_timestamp)
        p1_pings, p1_ts = list(pings[pings.provider == 'Voo'].rtt), list(pings[pings.provider == 'Voo'].capture_timestamp)
        assert p0_ts[0] < p0_ts[10] and p1_ts[0] < p1_ts[10] 
        net = TimeseriesNetwork([p0_pings, p1_pings], [p0_ts, p1_ts])
        ex = EXP3(2, max_weight=100, gamma=0.2)
        mh = MeanHistoryInternet({'a': TimeseriesNetwork([p0_pings, p1_pings], [p0_ts, p1_ts])}, history_len=10)
        oracle = OracleInternet({'a': TimeseriesNetwork([p0_pings, p1_pings], [p0_ts, p1_ts])})

        N_FLOWS = 1000
        flows_duration_range = (4, 100)
        max_rtt = 0
        flows: List[Flow] = []
        mh_flows: List[Flow] = []
        oracle_flows: List[Flow] = []
        last_v4 = float('inf')
        last_v6 = float('inf')
        for i in range(N_FLOWS):
            #if i / (N_FLOWS // 3) == 1:
            #    net.observations.reverse()
            #    mh.destinations['a'].observations.reverse()
            #if i / (N_FLOWS // 3) == 2:
            #    net.shifts[0] = -5 
            #    mh.destinations['a'].shifts[0] -5

            is_v6 = bool(ex.take_action())
            no_packets = random.randrange(*flows_duration_range)
            flow = NewRenoFlow(int(is_v6), packets=no_packets)
            while not flow.completed:
                flow.advance(net)
            reward = 1.0 if flow.smooth_rtt <= (last_v6 if flow.path_idx == 0 else last_v4) else 0
            if flow.path_idx == 0:
                last_v4 = flow.smooth_rtt
            else:
                last_v6 = flow.smooth_rtt

            mh_flow = mh.run_flow('a', no_packets)
            mh_flows.append(mh_flow)

            oracle_flow = oracle.run_flow('a', no_packets)
            oracle_flows.append(oracle_flow)
            
            #mean_v4 = [flow.smooth_rtt for flow in flows if flow.ip_version == 4][-100:]
            #mean_v4 = sum(mean_v4) / len(mean_v4) if mean_v4 else float('inf')
            #mean_v6 = [flow.smooth_rtt for flow in flows if flow.ip_version == 6][-100:]
            #mean_v6 = sum(mean_v6) / len(mean_v6) if mean_v6 else float('inf')
            #max_rtt = max(max_rtt, flow.smooth_rtt)
            #raw_reward = (max_rtt - flow.smooth_rtt) / max_rtt
            #reward = min(2 * (raw_reward ** 2), 1)
            #print(raw_reward, reward)
            #reward = 1.0 if flow.smooth_rtt < (mean_v6 if flow.ip_version == 4 else mean_v4) else 0
            ex.give_reward(int(is_v6),  reward)
            flows.append(flow)
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 9), sharex=True)
        ax_idx = 0

        for fls in [flows, mh_flows]:
            for pidx in [0, 1]:
                axes[0].step([i for i, f in enumerate(fls) if f.path_idx == pidx], 
                            [sum([f.path_idx == pidx for f in fls[:i]]) for i, f in enumerate(fls) if f.path_idx == pidx])
            
        axes[ax_idx].set_ylabel("Number of flows steered")
        axes[ax_idx].set_title("Flows in the network")
        axes[ax_idx].legend()
        ax_idx += 1

        for fls in [flows, mh_flows]:
            for pidx in [0, 1]:
                axes[ax_idx].scatter([idx for idx, f in enumerate(fls) if f.path_idx == pidx], [f.completion_time for idx, f in enumerate(fls) if f.path_idx == pidx], marker='.')        
        for fls in [flows, mh_flows, oracle_flows]:
            axes[ax_idx].plot([(sum(j.completion_time for j in fls[idx-10:idx]) / 10) for idx, f in enumerate(fls[10:])], label='Moving average (10f)')
        axes[ax_idx].set_ylim(40, 80)
        axes[ax_idx].set_ylabel('sRTT of the completed flow')
        axes[ax_idx].set_title("Flows RTT")
        axes[ax_idx].legend()
        ax_idx += 1

        for f1, f2 in zip(flows, oracle_flows):
            assert f2.completion_time <= f1.completion_time, (f1, f2)

        axes[ax_idx].plot([x for x in range(N_FLOWS)], [f2.completion_time - f1.completion_time for f1, f2 in zip(flows, mh_flows)], label='Mean history - EXP3')
        axes[ax_idx].plot([x for x in range(N_FLOWS)], [f2.completion_time - f1.completion_time for f1, f2 in zip(flows, oracle_flows)], label='Oracle - EXP3')
        #axes[ax_idx].hlines(y=0, xmin=0, xmax=N_FLOWS, color='lightgray')
        axes[ax_idx].set_ylim(-20, 20)
        axes[ax_idx].legend()
        ax_idx += 1

        axes[ax_idx].plot(list(ex._history.keys()), [ex._history[k]['probabilities'][0] for k in ex._history], label='P(v4)')
        axes[ax_idx].plot(list(ex._history.keys()), [ex._history[k]['probabilities'][1] for k in ex._history], label='P(v6)')
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].set_ylabel('Probability of a path')
        axes[ax_idx].set_title("EXP3 probabilities")
        axes[ax_idx].legend()
        ax_idx += 1

        axes[ax_idx].step([k for k in ex._history if ex._history[k]['action'] == 0], [ex._history[k]['reward'] for k in ex._history if ex._history[k]['action'] == 0], where='post', marker='.', label='IPv4')
        axes[ax_idx].step([k for k in ex._history if ex._history[k]['action'] == 1], [ex._history[k]['reward'] for k in ex._history if ex._history[k]['action'] == 1], where='post', marker='.', label='IPv6')
        axes[ax_idx].set_ylabel('EXP3 reward')
        axes[ax_idx].set_title("EXP3 reward")
        axes[ax_idx].legend()
        axes[ax_idx].set_xlabel("Flow # running in the network")
        ax_idx += 1

        plt.show()

        plt.figure()
        for fls in [flows, mh_flows, oracle_flows]:
            clts = sorted([f.completion_time for f in fls])
            plt.plot(clts, [x/len(clts) for x in range(len(clts))])
        plt.ylim(0, 1)
        plt.show()
    
    #test_single_destination()