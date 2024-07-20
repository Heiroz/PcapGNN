import torch
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import random
from decimal import Decimal
import torch.nn.functional as F

mappint_file_ids = 'output_ids.txt'
mapping_file_ports = 'output_ports.txt'
mapping_file_time = 'output_time_intervals.txt'

one_hot_class = 1024


def read_id_index_mapping(mapping_file):
    """
    从文件中读取第二个值映射到第一个值的映射关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 第二个值到第一个值的映射字典
    """
    id_index_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                index1 = int(parts[0])
                index2 = int(parts[1])
                id_index_mapping[index2] = index1
    return id_index_mapping



def map_id_to_index(id_val, id_index_mapping):
    """
    将id属性映射到正确的索引。
    如果有非法的映射，随机选择一个值从1到1000中给它，并打印出警告信息。
    
    :param id_val: int, id值
    :param id_index_mapping: dict, id到索引的映射字典
    :return: int, 对应的索引值
    """
    if id_val in id_index_mapping:
        return id_index_mapping[id_val]
    else:
        # 随机选择一个值从1到1000中给非法映射的id
        random_id_index = random.randint(1, 500)
        print(f"Warning: Invalid ID mapping for ID {id_val}. Assigned random ID index {random_id_index}.")
        return random_id_index



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


def map_port_to_index(port, port_index_mapping):
    """
    将端口属性映射到正确的索引。
    如果有非法的映射，随机选择一个值从1到500中给它，并打印出警告信息。
    
    :param port: int, 端口号
    :param port_index_mapping: dict, 端口号到索引的映射字典
    :return: int, 对应的索引值
    """
    if port in port_index_mapping:
        return port_index_mapping[port]
    else:
        # 随机选择一个值从1到500中给非法映射的端口
        random_port_index = random.randint(1, 500)
        print(f"Warning: Port {port} is not in the mapping. Assigning random index {random_port_index}.")
        return random_port_index


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


def map_time_to_index(time, time_index_mapping):
    """
    将时间戳映射到正确的索引。
    :param time: int, 时间戳
    :param time_index_mapping: dict, 时间戳区间映射到索引的映射字典
    :return: int, 对应的索引值
    """
    for time_interval, index in time_index_mapping.items():
        if time_interval[0] <= time <= time_interval[1]:
            return index
    return 1


def get_flows(filename):
    flows = extract_pcap_info(filename)
    flow_analysis, start_time, num_pkts = analyze_flows(flows)
    return flow_analysis, start_time, num_pkts


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
                'flag': int(ip_layer.flags),
                'time': float(pkt.time),
                'pkt_len': len(pkt),
            }

            flows[flow_key].append(packet_info)

    return flows


def onehot_encode(input_tensor, num_classes):
    input_tensor = input_tensor.long()
    one_hot_encoded = F.one_hot(input_tensor, num_classes=num_classes)
    flattened_one_hot = one_hot_encoded.view(-1)
    return flattened_one_hot

def decimal_to_10bit_binary(decimal_value):
    binary_str = bin(decimal_value)[2:]
    padded_binary_str = binary_str.zfill(10)
    binary_list = [int(bit) for bit in padded_binary_str]
    return binary_list

def convert_vector_to_binary(flow_vector):
    binary_representation = []
    for val in flow_vector:
        binary_representation.extend(decimal_to_10bit_binary(val.item()))
    return torch.tensor(binary_representation, dtype=torch.int)


def analyze_flows(flows):
    flow_analysis = []
    time_multiplier = 1  # 用于将time转换为整数的系数

    for flow_key, packets in flows.items():

        # 将flow_key中的元素转换为整数
        src_ip_features = [int(val) for val in flow_key[:4]]
        src_ip_features = torch.tensor(src_ip_features, dtype=torch.int)

        dst_ip_features = [int(val) for val in flow_key[4:8]]
        dst_ip_features = torch.tensor(dst_ip_features, dtype=torch.int)
        
        src_port = int(flow_key[8])
        port_index_mapping = read_port_index_mapping(mapping_file_ports)
        src_port = map_port_to_index(src_port, port_index_mapping)
        src_port = torch.tensor(src_port, dtype=torch.int).unsqueeze(0)

        dst_port = int(flow_key[9])
        port_index_mapping = read_port_index_mapping(mapping_file_ports)
        dst_port = map_port_to_index(dst_port, port_index_mapping)
        dst_port = torch.tensor(dst_port, dtype=torch.int).unsqueeze(0)

        protocol = int(flow_key[10])
        protocol = torch.tensor(protocol, dtype=torch.int).unsqueeze(0)

        start_time = int(min(pkt['time'] for pkt in packets) * time_multiplier)

        num_packets = len(packets)

        # 将所有字段连接在一起形成 flow_vector
        flow_vector = torch.cat([src_ip_features, dst_ip_features, src_port, dst_port, protocol])
        flow_vector = convert_vector_to_binary(flow_vector)
        # flow_vector = onehot_encode(flow_vector, 1024).to(torch.bool)

        remaining_features = []

        for pkt in packets:

            tos = int(pkt['tos'])

            ttl = int(pkt['ttl'])

            _id = int(pkt['id'])
            id_index_mapping = read_id_index_mapping(mappint_file_ids)
            _id = map_id_to_index(_id, id_index_mapping)

            flag = int(pkt['flag'])

            time = float(pkt['time']) * time_multiplier
            time_index_mapping = read_time_index_mapping(mapping_file_time)
            time = map_time_to_index(time, time_index_mapping)

            pkt_len = int(pkt['pkt_len'])
            
            pkt_tensor = torch.tensor([tos, ttl, _id, flag, time, pkt_len])
            pkt_tensor = convert_vector_to_binary(pkt_tensor)
            # pkt_tensor = onehot_encode(pkt_tensor, 1024).to(torch.bool)

            remaining_features.append(pkt_tensor)

        remaining_features = torch.stack(remaining_features, dim=0)

        flow_info = {
            'flow_vector': flow_vector,
            'remaining_features': remaining_features
        }

        flow_analysis.append(flow_info)

    return flow_analysis, start_time, num_packets