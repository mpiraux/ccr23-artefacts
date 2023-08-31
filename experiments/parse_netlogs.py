#!/usr/bin/env python
import os
import sys
import ujson as json
import pandas as pd
import zipfile
from tqdm import tqdm

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} result_dir output_json")
    exit(-1)


def canonical_ip(ip_str: str) -> str:
        import ipaddress
        if '[' in ip_str:
            ip_str = ip_str[1:ip_str.rindex(']')]
        elif '.' in ip_str and ':' in ip_str and ip_str.index('.') < ip_str.index(':'):
            assert ip_str.count(':') == 1
            ip_str = ip_str[:ip_str.index(':')]
        a = ipaddress.ip_address(ip_str)
        if a.version == 6 and a.ipv4_mapped:
            return str(a.ipv4_mapped)
        return str(a)


result_dir = sys.argv[1]
data = []
for d in tqdm(os.listdir(result_dir)):
    if not os.path.isdir(os.path.join(result_dir, d)):
        continue
    for m in os.listdir(os.path.join(result_dir, d)):
        for r in os.listdir(os.path.join(result_dir, d, m)):
            run_dir = os.path.join(result_dir, d, m , r)
            if os.path.exists(os.path.join(run_dir, 'netlog.zip')):
                with zipfile.ZipFile(os.path.join(run_dir, 'netlog.zip')) as zf:
                    netlog_str = zf.read(zf.filelist[0]).decode()
            else:
                with open(os.path.join(run_dir, 'netlog.json')) as f:
                    netlog_str = f.read()
            try:
                netlog = json.loads(netlog_str)
            except json.JSONDecodeError:
                if netlog_str.endswith('},\n'):
                    netlog_str = netlog_str[:-2] + ']}\n'
                elif netlog_str.endswith('}'):
                    netlog_str += ']}\n'
                try:
                    netlog = json.loads(netlog_str)
                except json.JSONDecodeError:
                    print(run_dir, "failed to parse netlog")
                    break
            constants = netlog['constants']
            for k, v in constants.items():
                if type(v) is dict:
                    constants[k] = dict((v, k) for k, v in v.items())
            events = netlog['events']
            source_events = {}
            for e in events:
                e['type'] = constants['logEventTypes'][e['type']]
                e['phase'] = constants['logEventPhase'][e['phase']]
                e['source']['type'] = constants['logSourceType'][e['source']['type']]
                se = source_events.get(e['source']['id'], e['source'])
                l = se.get('events', [])
                del e['source']
                l.append(e)
                se['events'] = l
                source_events[se['id']] = se

            rdns = {}
            for se in source_events.values():
                if se['type'] == 'HOST_RESOLVER_IMPL_JOB':
                    host = None
                    ips = []
                    for e in se['events']:
                        if e['phase'] == 'PHASE_BEGIN' and e['type'] == 'HOST_RESOLVER_MANAGER_JOB':
                            host = e['params']['host']
                        if e['phase'] == 'PHASE_END' and e['type'] == 'HOST_RESOLVER_SYSTEM_TASK' and 'params' in e:
                            ips = e['params'].get('address_list')
                        if e['phase'] == 'PHASE_END' and e['type'] == 'HOST_RESOLVER_MANAGER_DNS_TASK' and 'params' in e:
                            ips = list(map(lambda x: x['endpoint_address'], e['params'].get('results', {}).get('ip_endpoints', [])))
                    assert host
                    if not ips:
                        continue
                    for ip in ips:
                        ip = canonical_ip(ip)
                        rdns[ip] = host.replace('http://', '').replace('https://', '')

            tcp_connections_time = []
            quic_connections_time = []
            for se in source_events.values():
                if se['type'] == 'SOCKET':
                    start_time = 0
                    address = None
                    for e in se['events']:
                        if e['phase'] == 'PHASE_BEGIN' and e['type'] == 'TCP_CONNECT':
                            start_time = int(e['time'])
                            assert len(e['params']['address_list']) == 1
                            address = e['params']['address_list'][0]
                        elif e['phase'] == 'PHASE_END' and e['type'] == 'TCP_CONNECT':
                            assert start_time
                            if 'remote_address' not in e['params']:
                                continue
                            address = None
                            tcp_connections_time.append((e['params']['remote_address'], int(e['time']) - start_time))
                            break
                    if address:
                        tcp_connections_time.append((address, float('nan')))
                elif se['type'] == 'QUIC_SESSION':
                    start_time = 0
                    remote_address = None
                    for e in se['events']:
                        if e['type'] == 'QUIC_SESSION_PACKET_SENT' and start_time == 0:
                            start_time = int(e['time'])
                        elif e['type'] == 'QUIC_SESSION_PACKET_RECEIVED':
                            assert start_time
                            remote_address = e['params']['peer_address']
                            quic_connections_time.append((remote_address, int(e['time']) - start_time))
                            break

            for ip, conn_time in tcp_connections_time:
                ip = canonical_ip(ip)
                if ip not in rdns:
                    print(d, m, r)
                    continue
                data.append((d, m, r, rdns[ip], ip, 'tcp', conn_time))
            for ip, conn_time in quic_connections_time:
                ip = canonical_ip(ip)
                if ip not in rdns:
                    print(d, m, r)
                    continue
                data.append((d, m, r, rdns[ip], ip, 'quic', conn_time))

pd.DataFrame(data, columns=['page', 'mode', 'run', 'domain', 'ip', 'proto', 'conn_time']).to_json(sys.argv[2])