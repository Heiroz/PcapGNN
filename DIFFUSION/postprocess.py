import pandas as pd
import random

def read_index_port_mapping(mapping_file):
    """
    从文件中读取索引映射回端口号的关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 索引到端口号的映射字典
    """
    index_port_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                index1 = int(parts[0])
                index2 = int(parts[1])
                index_port_mapping[index1] = index2
    return index_port_mapping

def map_indices_to_ports(port_index_mapping):
    """
    将端口索引映射回原始端口号。
    :param port_index_mapping: dict, 第二个值到第一个值的映射字典
    :return: dict, 端口索引映射回原始端口号的映射字典
    """
    index_port_mapping = {v: k for k, v in port_index_mapping.items()}
    return index_port_mapping

def read_index_timestamp_mapping(mapping_file):
    """
    从文件中读取索引映射回时间戳的关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 索引到时间戳的映射字典
    """
    index_timestamp_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                index = int(parts[0])
                timestamp1 = float(parts[1])
                timestamp2 = float(parts[2])
                # 随机选择一个时间戳映射回原始时间戳
                mapped_timestamp = random.uniform(timestamp1, timestamp2)
                index_timestamp_mapping[index] = mapped_timestamp
    return index_timestamp_mapping

def map_indices_to_timestamps(timestamp_index_mapping):
    """
    将时间戳索引映射回原始时间戳。
    :param timestamp_index_mapping: dict, 第一个值到第二个值和第三个值的映射字典
    :return: dict, 时间戳索引映射回原始时间戳的映射字典
    """
    index_timestamp_mapping = {k: v for k, v in timestamp_index_mapping.items()}
    return index_timestamp_mapping

def construct_table(sample_graphs, filename, port_index_mapping_file, timestamp_index_mapping_file):
    all_tables = []

    # 读取端口索引映射文件
    port_index_mapping = read_index_port_mapping(port_index_mapping_file)
    # 读取时间戳索引映射文件
    timestamp_index_mapping = read_index_timestamp_mapping(timestamp_index_mapping_file)

    for graph in sample_graphs:
        node_attrs = graph[0]  # (n, node_attr_dim)
        edge_attrs = graph[1]  # (n, n, edge_attr_dim)

        n = node_attrs.shape[0]

        for i in range(n):
            src_ip = ".".join(map(str, node_attrs[i, :4].tolist()))  # 前四个值为源IP地址
            src_port = node_attrs[i, 4].item()  # 第五个值为源端口

            for j in range(n):
                if i != j:
                    protocol = edge_attrs[i, j, 0].item()  # 第一个值为协议
                    pkt_count = edge_attrs[i, j, 1].item()  # 第二个值为包个数
                    timestamp_index = edge_attrs[i, j, 2].item()  # 第三个值为时间戳索引

                    if protocol == 0:
                        continue
                    
                    # 将dst_ip和dst_port使用node_attrs[j]中的数据
                    dst_ip = ".".join(map(str, node_attrs[j, :4].tolist()))  # 前四个值为目的IP地址
                    dst_port = node_attrs[j, 4].item()  # 第五个值为目的端口

                    # 检查src_port和dst_port是否在端口映射中
                    if src_port not in port_index_mapping or dst_port not in port_index_mapping:
                        continue

                    # 检查timestamp_index是否在时间戳映射中
                    if timestamp_index not in timestamp_index_mapping:
                        continue

                    # 将端口索引映射回原始端口号
                    src_port_mapped = port_index_mapping[src_port]
                    dst_port_mapped = port_index_mapping[dst_port]

                    # 将时间戳索引映射回原始时间戳
                    timestamp = timestamp_index_mapping[timestamp_index]

                    row = [src_ip, dst_ip, src_port_mapped, dst_port_mapped, protocol, pkt_count, timestamp]
                    all_tables.append(row)

    columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'pkt_count', 'timestamp']

    df = pd.DataFrame(all_tables, columns=columns)
    df.to_csv(filename, index=False)

    return
