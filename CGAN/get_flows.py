import torch
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import random

def get_flows(filename):
    flows = extract_pcap_info(filename)
    flow_analysis = analyze_flows_decode(flows)
    return flow_analysis

def ip_to_features(ip):
    return [int(octet) for octet in ip.split('.')]


def process_flow_data(file_path):
    """
    读取CSV文件并将每行数据转换为特征向量。
    :param file_path: str, CSV文件路径
    :return: torch.Tensor, 所有流量特征向量组成的张量
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 初始化列表以存储所有流向量
    flow_vectors = []

    # 遍历每一行数据
    for index, row in df.iterrows():
        src_ip_features = ip_to_features(row['src_ip'])
        dst_ip_features = ip_to_features(row['dst_ip'])
        src_port = torch.tensor([0])
        dst_port = torch.tensor([0])
        protocol = torch.tensor([row['protocol']])
        start_time = torch.tensor([random.random()])
        
        # 合并所有特征为一个向量
        flow_vector = torch.cat([src_ip_features, dst_ip_features, src_port, dst_port, protocol, start_time])
        flow_vectors.append(flow_vector)

    # 转换为tensor
    flow_vectors = torch.stack(flow_vectors)
    
    return flow_vectors


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
                'start_time': float(pkt.time),
                'pkt_len': len(pkt),
            }

            flows[flow_key].append(packet_info)

    return flows

def convert_to_binary_vector(value, bit_length=16):
    binary_str = bin(value)[2:]
    binary_str = binary_str.zfill(bit_length)
    binary_vector = torch.tensor([int(bit) for bit in binary_str], dtype=torch.float32)
    return binary_vector

import torch

def analyze_flows(flows):
    flow_analysis = []
    start_time_multiplier = 1  # 用于将start_time转换为整数的系数

    for flow_key, packets in flows.items():
        # 将flow_key中的元素转换为整数
        src_ip_features = [int(val) for val in flow_key[:4]]
        dst_ip_features = [int(val) for val in flow_key[4:8]]
        src_port = int(flow_key[8])
        dst_port = int(flow_key[9])
        protocol = int(flow_key[10])
        start_time = int(min(pkt['start_time'] for pkt in packets) * start_time_multiplier)
        num_packets = len(packets)

        # 将所有字段连接在一起形成 flow_vector
        flow_vector = torch.tensor(src_ip_features + dst_ip_features + [src_port, dst_port, protocol, start_time])

        remaining_features = []

        for pkt in packets:
            tos = int(pkt['tos'])
            ttl = int(pkt['ttl'])
            _id = int(pkt['id'])
            flag = int(pkt['flag'])
            start_time = float(pkt['start_time']) * start_time_multiplier
            pkt_len = int(pkt['pkt_len'])

            remaining_features.append(torch.tensor([tos, ttl, _id, flag, start_time, pkt_len]))

        remaining_features = torch.stack(remaining_features, dim=0)

        flow_info = {
            'flow_vector': flow_vector,
            'remaining_features': remaining_features
        }

        flow_analysis.append(flow_info)

    return flow_analysis


def analyze_flows_decode(flows):
    flow_analysis = []
    bit_length = 32  # 定义固定长度的二进制序列
    start_time_multiplier = 1  # 用于将start_time转换为整数的系数

    for flow_key, packets in flows.items():
        # 将flow_key中的元素转换为整数
        src_ip_features = [int(val) for val in flow_key[:4]]
        dst_ip_features = [int(val) for val in flow_key[4:8]]
        src_port = int(flow_key[8])
        dst_port = int(flow_key[9])
        protocol = int(flow_key[10])
        start_time = int(min(pkt['start_time'] for pkt in packets) * start_time_multiplier)
        num_packets = int(len(packets))

        # 将字段转换为二进制向量
        src_ip_features = torch.cat([convert_to_binary_vector(val, bit_length) for val in src_ip_features])
        dst_ip_features = torch.cat([convert_to_binary_vector(val, bit_length) for val in dst_ip_features])
        src_port = convert_to_binary_vector(src_port, bit_length)
        dst_port = convert_to_binary_vector(dst_port, bit_length)
        protocol = convert_to_binary_vector(protocol, bit_length)
        start_time = convert_to_binary_vector(start_time, bit_length)
        num_packets = torch.tensor([num_packets])
        # print(start_time.size())
        # 将所有二进制序列连接在一起形成 flow_vector
        flow_vector = torch.cat([src_ip_features, dst_ip_features, src_port, dst_port, protocol, start_time])

        remaining_features = []

        for pkt in packets:
            tos = convert_to_binary_vector(int(pkt['tos']), bit_length)
            ttl = convert_to_binary_vector(int(pkt['ttl']), bit_length)
            _id = convert_to_binary_vector(int(pkt['id']), bit_length)
            flag = convert_to_binary_vector(int(pkt['flag']), bit_length)
            start_time = convert_to_binary_vector(int(float(pkt['start_time']) * start_time_multiplier), bit_length)
            pkt_len = convert_to_binary_vector(int(pkt['pkt_len']), bit_length)

            remaining_features.append(torch.cat([tos, ttl, _id, flag, start_time, pkt_len]))

        remaining_features = torch.stack(remaining_features, dim=0)

        flow_info = {
            'flow_vector': flow_vector,
            'remaining_features': remaining_features
        }

        flow_analysis.append(flow_info)

    return flow_analysis
