from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from decimal import Decimal
import random

def get_flows(filename):
    flows = extract_pcap_info(filename)
    flow_analysis = analyze_flows(flows)
    return flow_analysis

def ip_to_features(ip):
    return [int(octet) for octet in ip.split('.')]

def extract_pcap_info(pcap_file):
    packets = rdpcap(pcap_file)
    flows = defaultdict(list)

    for pkt in packets:
        if IP in pkt:
            ip_layer = pkt[IP]

            if TCP in pkt:
                transport_layer = pkt[TCP]
            elif UDP in pkt:
                transport_layer = pkt[UDP]
            else:
                continue

            src_ip_features = ip_to_features(ip_layer.src)
            dst_ip_features = ip_to_features(ip_layer.dst)
            flow_key = (*src_ip_features, *dst_ip_features, transport_layer.sport, transport_layer.dport, ip_layer.proto)

            packet_info = {
                'tos': ip_layer.tos,
                'ttl': ip_layer.ttl,
                'id': ip_layer.id,
                'flag': ip_layer.flags,
                'pkt_len': len(pkt),
                'time': pkt.time
            }

            flows[flow_key].append(packet_info)

    return flows

def analyze_flows(flows):
    flow_analysis = []

    for flow_key, packets in flows.items():
        src_ip_features = flow_key[:4]
        dst_ip_features = flow_key[4:8]
        src_port, dst_port, protocol = flow_key[8:11]
        time = min(pkt['time'] for pkt in packets)
        num_packets = len(packets)
        
        remaining_features = {
            'tos': [pkt['tos'] for pkt in packets],
            'ttl': [pkt['ttl'] for pkt in packets],
            'id': [pkt['id'] for pkt in packets],
            'flag': [pkt['flag'] for pkt in packets],
            'pkt_len': [pkt['pkt_len'] for pkt in packets],
            'time': [pkt['time'] for pkt in packets]
        }

        flow_vector = list(src_ip_features) + list(dst_ip_features) + [src_port, dst_port, protocol, time, num_packets]

        flow_info = {
            'flow_vector': flow_vector,
            'remaining_features': remaining_features
        }

        flow_analysis.append(flow_info)

    return flow_analysis

def classify_pcap_split(pcap_file, split_count=200):
    packets = rdpcap(pcap_file)
    total_packets = len(packets)
    packets_per_split = total_packets // split_count
    ip_pairs_list = []

    for i in range(split_count):
        start_idx = i * packets_per_split
        end_idx = (i + 1) * packets_per_split if i < split_count - 1 else total_packets
        split_packets = packets[start_idx:end_idx]
        ip_pairs = {}

        for pkt in split_packets:
            if pkt.haslayer(IP):
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                protocol = pkt[IP].proto
                src_port, dst_port = None, None

                if pkt.haslayer(TCP):
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                elif pkt.haslayer(UDP):
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport

                ip_pair = (src_ip, dst_ip, src_port, dst_port, protocol)

                if ip_pair not in ip_pairs:
                    ip_pairs[ip_pair] = {
                        'time': None,
                        'end_time': None,
                        'duration': 0,
                        'flow_len': 0
                    }

                pkt_time = pkt.time
                if ip_pairs[ip_pair]['time'] is None or pkt_time < ip_pairs[ip_pair]['time']:
                    ip_pairs[ip_pair]['time'] = pkt_time
                if ip_pairs[ip_pair]['end_time'] is None or pkt_time > ip_pairs[ip_pair]['end_time']:
                    ip_pairs[ip_pair]['end_time'] = pkt_time

                ip_pairs[ip_pair]['flow_len'] += 1
        
        for ip_pair in ip_pairs:
            time = ip_pairs[ip_pair]['time']
            end_time = ip_pairs[ip_pair]['end_time']
            duration = end_time - time if time and end_time else 0
            ip_pairs[ip_pair]['duration'] = duration

        ip_pairs_list.append(ip_pairs)

    return ip_pairs_list

def pcap2graph(ip_pairs):
    G = nx.DiGraph()
    ip_port_to_id = {}
    current_id = 0
    
    for ip_pair, info in ip_pairs.items():
        src_ip, dst_ip, src_port, dst_port, protocol = ip_pair
        flow_len = info['flow_len']
        time = info['time']
        duration = info['duration']
        
        src_key = (src_ip, src_port)
        dst_key = (dst_ip, dst_port)
        
        if src_key not in ip_port_to_id:
            ip_port_to_id[src_key] = current_id
            current_id += 1
            G.add_node(ip_port_to_id[src_key], ip=src_ip, port=src_port)
            
        if dst_key not in ip_port_to_id:
            ip_port_to_id[dst_key] = current_id
            current_id += 1
            G.add_node(ip_port_to_id[dst_key], ip=dst_ip, port=dst_port)
            
        G.add_edge(ip_port_to_id[src_key], ip_port_to_id[dst_key], protocol=protocol, 
                   flow_len=flow_len, time=time, duration=duration)
        
    return G


def edge_attr(G):
    protocol_attrs = nx.get_edge_attributes(G, 'protocol')
    flow_len_attrs = nx.get_edge_attributes(G, 'flow_len')
    time_attrs = nx.get_edge_attributes(G, 'time')
    mapping_file = 'time_index_mapping.txt'
    time_index_mapping = read_time_index_mapping(mapping_file)
    time_attrs = map_start_time_to_indices(time_attrs, time_index_mapping)
    duration_attrs = nx.get_edge_attributes(G, 'duration')
    
    num_edges = G.number_of_edges()
    num_attributes = 3
    
    edge_attr_tensor = np.zeros((num_edges, num_attributes))
    edge_index = np.zeros((2, num_edges), dtype=int)
    
    for idx, (u, v) in enumerate(G.edges()):
        edge_index[0, idx] = u
        edge_index[1, idx] = v
        edge_attr_tensor[idx, 0] = protocol_attrs[(u, v)]
        edge_attr_tensor[idx, 1] = flow_len_attrs[(u, v)]
        edge_attr_tensor[idx, 2] = time_attrs[(u, v)]
        # edge_attr_tensor[idx, 3] = duration_attrs[(u, v)]
    
    edge_attr_tensor = torch.tensor(edge_attr_tensor, dtype=torch.long)
    edge_attr_tensor = onehot_encode(num_attributes, edge_attr_tensor)
    return edge_attr_tensor, edge_index

import networkx as nx


def read_time_index_mapping(mapping_file):
    """
    从文件中读取时间戳区间映射到索引的映射关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 时间戳区间映射到索引的映射字典
    """
    time_index_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                index = int(parts[0])
                time1 = Decimal(parts[1])
                time2 = Decimal(parts[2])
                time_index_mapping[(time1, time2)] = index
    return time_index_mapping

def map_start_time_to_indices(start_time_attrs, time_index_mapping):
    """
    将时间戳区间映射到正确的索引。
    :param start_time_attrs: dict, 时间戳区间属性字典
    :param time_index_mapping: dict, 时间戳区间映射到索引的映射字典
    :return: dict, 更新后的时间戳索引属性字典
    """
    updated_start_time_attrs = {}
    for edge, start_time in start_time_attrs.items():
        mapped_index = None
        for time_interval, index in time_index_mapping.items():
            if time_interval[0] <= start_time <= time_interval[1]:
                mapped_index = index
                break
        updated_start_time_attrs[edge] = mapped_index
    
    return updated_start_time_attrs

def read_port_index_mapping(mapping_file):
    """
    从文件中读取第二个值映射到第一个值的映射关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 第二个值到第一个值的映射字典
    """
    port_index_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                index1 = int(parts[0])
                index2 = int(parts[1])
                port_index_mapping[index2] = index1
    return port_index_mapping

def map_ports_to_indices(G, port_attrs, port_index_mapping):
    """
    将图G中的端口属性映射到正确的索引，并更新到图的节点属性中。
    如果有非法的映射，随机选择一个值从1到500中给它，并打印出警告信息。
    
    :param G: networkx.Graph, 图对象
    :param port_attrs: dict, 端口属性字典
    :param port_index_mapping: dict, 第二个值到第一个值的映射字典
    :return: dict, 更新后的端口索引属性字典
    """
    updated_port_attrs = {}
    invalid_mappings = []  # 存储非法映射的列表
    for node, port in port_attrs.items():
        if port in port_index_mapping:
            updated_port_attrs[node] = port_index_mapping[port]
            nx.set_node_attributes(G, {node: {'port': port_index_mapping[port]}})
        else:
            # 随机选择一个值从1到500中给非法映射的端口
            random_port_index = random.randint(1, 500)
            updated_port_attrs[node] = random_port_index
            nx.set_node_attributes(G, {node: {'port': random_port_index}})
            invalid_mappings.append((node, port))
    
    return updated_port_attrs


def ip2ints(ip):
    return list(map(int, ip.split('.')))

def node_attr(G):
    ip_attrs = nx.get_node_attributes(G, 'ip')
    port_attrs = nx.get_node_attributes(G, 'port')
    # ###################################################################################
    mapping_file = 'port_index_mapping.txt'
    port_index_mapping = read_port_index_mapping(mapping_file)
    port_attrs = map_ports_to_indices(G, port_attrs, port_index_mapping)
    num_nodes = G.number_of_nodes()
    num_ip_attributes = 4
    num_port_attributes = 1
    
    node_attr_tensor = np.zeros((num_nodes, num_ip_attributes + num_port_attributes))
    
    for idx, node in enumerate(G.nodes()):
        ip_parts = ip2ints(ip_attrs[node])
        port_part = [port_attrs[node]]
        combined_attr = np.concatenate((ip_parts, port_part))
        node_attr_tensor[idx, :] = combined_attr
    
    node_attr_tensor = torch.tensor(node_attr_tensor, dtype=torch.long)
    node_attr_tensor = onehot_encode(num_ip_attributes + num_port_attributes, node_attr_tensor)
    return node_attr_tensor

def onehot_encode(n_attr, attr_tensor):
    one_hot_encoded_attrs = []
    for i in range(n_attr):
        one_hot = F.one_hot(attr_tensor[:, i], num_classes=1024).float()
        one_hot_encoded_attrs.append(one_hot)
    encoded_attr_tensor = torch.cat(one_hot_encoded_attrs, dim=1)
    return encoded_attr_tensor

def adjacency_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).toarray()
    return np.array(adj_matrix)
