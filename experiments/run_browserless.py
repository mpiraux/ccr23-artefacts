#!/usr/bin/env python3

import subprocess
import time
import socket
import threading
import os
import datetime
import shutil
import random
import zipfile

domains = """google.com
facebook.com
youtube.com
cloudflare.com
instagram.com
apple.com
linkedin.com
wikipedia.org
netflix.com
yahoo.com
googletagmanager.com
bing.com
doubleclick.net
office.com
googlevideo.com
reddit.com
whatsapp.com
youtu.be
zoom.us
mail.ru
adobe.com
yandex.ru
goo.gl
taobao.com
googledomains.com
google-analytics.com
blogspot.com
fbcdn.net
spotify.com
icloud.com
myfritz.net
canva.com
europa.eu
google.com.hk
dropbox.com
yandex.net
medium.com
t.me
lencr.org
apache.org
""".splitlines()

IP = 'TODO'
FORWARD_IP = '127.0.0.1'
FORWARD_TO = 1053

def run(command) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, universal_newlines=True, shell=True)

def udp_relay():
    TIMEOUT_SECONDS = 10
    LISTEN_ON = 53

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((IP, LISTEN_ON))
    except OSError:
        print("Unable to bind on port ", LISTEN_ON)
        exit(0)

    sock_dict = { }

    # Lock.
    lock = threading.Lock()

    # One thread for each connection.
    class ListenThread(threading.Thread):
        def __init__(self, info):
            threading.Thread.__init__(self)
            self.s_client = info['socket']
            self.s_client.settimeout(1)
            self.addr = info['addr']
            self.last_receive = time.time()
            self.should_stop = False

        def run(self):
            while not self.should_stop:
                try: data, r_addr = self.s_client.recvfrom(65536)
                except:
                    if time.time() - self.last_receive > TIMEOUT_SECONDS:
                        break
                    else:
                        continue
                # Reset timeout.
                self.last_receive = time.time()
                # Successfully received a packet, forward it.
                sock.sendto(data, self.addr)
            lock.acquire()
            try:
                self.s_client.close()
                sock_dict.pop(self.addr)
            except: pass
            lock.release()

        def stop(self):
            self.should_stop = True

    try:
        while True:
            data, addr = sock.recvfrom(65536)
            lock.acquire()
            try:
                if not addr in sock_dict:
                    s_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    item = {
                        "socket": s_client,
                        "addr": addr
                    }
                    s_client.sendto(data, (FORWARD_IP, FORWARD_TO))
                    t = ListenThread(item)
                    t.start()
                    item['thread'] = t
                    sock_dict[addr] = item
                else:
                    s_client = sock_dict[addr]['socket']
                    s_client.sendto(data, (FORWARD_IP, FORWARD_TO))
            except: pass
            lock.release()
    except: pass

    # Stop all threads.
    for addr in sock_dict:
        try: sock_dict[addr]['thread'].stop()
        except: pass

def parse_netlog(run_dir: str) -> [(str, int, int)]:
    def canonical_ip(ip_str: str) -> str:
        import ipaddress
        if '[' in ip_str:
            ip_str = ip_str[1:ip_str.rindex(']')]
        a = ipaddress.ip_address(ip_str)
        if a.version == 6 and a.ipv4_mapped:
            return str(a.ipv4_mapped)
        return str(a)

    import json
    data = []
    with open(os.path.join(run_dir, 'netlog.json')) as f:
        try:
            netlog = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            json_str = f.read()
            if json_str.endswith('},\n'):
                json_str = json_str[:-2] + ']}\n'
            elif json_str.endswith('}'):
                json_str += ']}\n'
            try:
                netlog = json.loads(json_str)
            except json.JSONDecodeError:
                print(run_dir, "failed to parse netlog")
                return
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

    tcp_connections_time = {}
    quic_connections_time = {}
    for se in source_events.values():
        if se['type'] == 'SOCKET':
            start_time = 0
            for e in se['events']:
                if e['phase'] == 'PHASE_BEGIN' and e['type'] == 'TCP_CONNECT':
                    start_time = int(e['time'])
                elif e['phase'] == 'PHASE_END' and e['type'] == 'TCP_CONNECT':
                    assert start_time
                    if 'remote_address' not in e['params']:
                        continue
                    tcp_connections_time[e['params']['remote_address']] = int(e['time']) - start_time
                    break
        elif se['type'] == 'QUIC_SESSION':
            start_time = 0
            remote_address = None
            for e in se['events']:
                if e['type'] == 'QUIC_SESSION_PACKET_SENT' and start_time == 0:
                    start_time = int(e['time'])
                elif e['type'] == 'QUIC_SESSION_PACKET_RECEIVED':
                    assert start_time
                    remote_address = e['params']['peer_address']
                    quic_connections_time[remote_address] = int(e['time']) - start_time
                    break

    for ip, conn_time in tcp_connections_time.items():
        ip = canonical_ip(ip)
        data.append((rdns[ip], 6 if ':' in ip else 4, conn_time))
    for ip, conn_time in quic_connections_time.items():
        ip = canonical_ip(ip)
        data.append((rdns[ip], 6 if ':' in ip else 4, conn_time))
    return data

def pack_udp_feedback(feedback: [(str, int, int)]):
    import struct
    packet = struct.pack('!B', len(feedback))
    for domain, ipv, ctime in feedback:
        assert len(domain) < 255
        domain_bytes = domain.encode()
        packet += struct.pack(f'!{len(domain_bytes)+1}pBH', domain_bytes, ipv, ctime)

    return packet

with open(os.path.join(os.path.dirname(__file__), 'config')) as f:
    for l in f:
        if l.startswith('proxy'):
            resolver_ip, resolver_port = l.split()[1].split(':')
            resolver_port = int(resolver_port)

container_stop = "docker container rm -f {}"
container_logs = "docker logs {}"
container_cp = "docker cp {}:{} {}"
container_exec = "docker container exec -it {} {}"

updns_cid = "docker container ls -f ancestor=updns -q"
updns_cid2 = "docker container ls -f ancestor=updns -q"  # If you're using podman-docker, you need to change ancestor=updns to ancestor=localhost/updns
updns_start = "docker run -d -p 127.0.0.1:{hport}:53/udp -p 127.0.0.1:{hfport}:50000/udp -v ./config:/config -it updns -c /config run {mode}"

browserless_start = f"docker run -d --dns {IP} --network host -p 3000:3000 -v /tmp/logs/:/logs/ docker.io/browserless/chrome"
browserless_run = "node index.js https://{}/"
browserless_cid = "docker container ls -f ancestor=browserless/chrome -q"  # If you're using podman-docker, you need to change ancestor=browserless/chrome to ancestor=docker.io/browserless/chrome

print("IP is", IP, ", resolver is", resolver_ip, 'on port', resolver_port)

if __name__ == "__main__":
    for cid in run(updns_cid).stdout.strip().splitlines():
        print("Stopping previous updns instance", cid)
        run(container_stop.format(cid))

    for cid in run(updns_cid2).stdout.strip().splitlines():
        print("Stopping previous updns instance", cid)
        run(container_stop.format(cid))

    for cid in run(browserless_cid).stdout.strip().splitlines():
        print("Stopping previous browserless instance", cid)
        run(container_stop.format(cid))

    print("Checking for available ports: ", end='')
    for p in [53, 1053, 1054, 1056, 50000, 50004, 50006]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((IP, p))
            sock.close()
        except OSError:
            print(f"Cannot bind to {IP}:{p}, updns container will fail")
            exit(0)
    print("Ok")

    feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback_sock.connect(('127.0.0.1', 50000))

    print("Starting UDP relay")
    threading.Thread(target=udp_relay, daemon=True).start()

    print("Starting browserless")
    os.makedirs('/tmp/logs/', exist_ok=True)
    bcid = run(browserless_start).stdout.strip()
    assert bcid
    time.sleep(2)

    experiments_dir = os.path.join('results', datetime.datetime.now().isoformat().replace('.', ':').replace(':', '-'))
    os.makedirs(experiments_dir)

    random.seed("run_browserless")
    parameters = []

    for d in domains:
        for r in range(30):
            parameters.append((d, r))
    random.shuffle(parameters)

    modes_to_addresses = {'exp3': ('127.0.0.1', 1053), 'v4': ('127.0.0.1', 1054), 'v6': ('127.0.0.1', 1056), 'he': (resolver_ip, resolver_port)}

    ucid = run(updns_start.format(hport=modes_to_addresses['exp3'], hfport=50000, mode='exp3')).stdout.strip()
    assert ucid
    ucid_v4 = run(updns_start.format(hport=modes_to_addresses['v4'], hfport=50004, mode='v4')).stdout.strip()
    assert ucid_v4
    ucid_v6 = run(updns_start.format(hport=modes_to_addresses['v6'], hfport=50006, mode='v6')).stdout.strip()
    assert ucid_v6
    time.sleep(2)

    for d, r in parameters:
        modes = ['exp3', 'v4', 'v6', 'he']
        random.shuffle(modes)
        for m in modes:
            FORWARD_TO = modes_to_addresses[m]
            print("Starting experiment", d, m, "run", r)
            run_dir = os.path.join(experiments_dir, d, m, f'run_{r}')
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, 'browser.json'), 'w') as f:
                brun = run(browserless_run.format(d))
                if not brun.stdout.strip():
                    print(brun)
                f.write(brun.stdout)
            
            run(container_exec.format(bcid, '/usr/bin/sync'))
            run(container_cp.format(bcid, f"/tmp/netlog.json", os.path.join(run_dir, 'netlog.json')))
            run(container_cp.format(bcid, f"/tmp/key.log", os.path.join(run_dir, 'key.log')))
            if m == 'exp3':
                feedback = pack_udp_feedback(parse_netlog(run_dir))
                feedback_sock.send(feedback)
            with zipfile.ZipFile(os.path.join(run_dir, 'netlog.zip'), mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                zf.write(os.path.join(run_dir, 'netlog.json'), arcname='netlog.json')
            os.unlink(os.path.join(run_dir, 'netlog.json'))
            run(container_exec.format(bcid, 'rm -f /tmp/key.log'))
    for u, fn in zip([ucid, ucid_v4, ucid_v6], ['updns.log', 'updns_v4.log', 'updns_v6.log']):
        with open(os.path.join(experiments_dir, fn), 'w') as f:
            f.write(run(container_logs.format(u)).stdout)
        run(container_stop.format(u))
    run(container_stop.format(bcid))
