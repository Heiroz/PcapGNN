from scapy.all import rdpcap
from collections import Counter
index_num = 576
def extract_ports_from_pcap(file_path):
    """
    读取PCAP文件并提取所有合法的端口号。
    :param file_path: str, PCAP文件路径
    :return: list, 包含所有合法端口号的列表
    """
    packets = rdpcap(file_path)
    ports = set()

    for packet in packets:
        if packet.haslayer('TCP') or packet.haslayer('UDP'):
            if packet.haslayer('TCP'):
                src_port = packet['TCP'].sport
                dst_port = packet['TCP'].dport
            elif packet.haslayer('UDP'):
                src_port = packet['UDP'].sport
                dst_port = packet['UDP'].dport
            
            # 添加源端口和目标端口到集合
            ports.update([src_port, dst_port])

    return list(ports)

def extract_time_intervals_from_pcap(file_path):
    """
    获取PCAP文件中数据包的时间范围，并将时间戳映射为区间索引。
    :param file_path: str, PCAP文件路径
    :return: list, 包含所有数据包时间戳对应的区间索引
    """
    packets = rdpcap(file_path)
    timestamps = [packet.time for packet in packets]

    min_time = min(timestamps)
    max_time = max(timestamps)
    interval_size = (max_time - min_time) / index_num  # 计算每个区间的大小

    index_mapping = []
    for timestamp in timestamps:
        index = int((timestamp - min_time) // interval_size)
        index_mapping.append(index)

    return index_mapping, min_time, max_time

def save_to_file(data, output_file):
    """
    将数据保存到文件中。
    :param data: list, 要保存的数据列表
    :param output_file: str, 输出文件路径
    """
    with open(output_file, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

if __name__ == "__main__":
    file_path = 'caida_small.pcap'  # 请替换为你的PCAP文件路径
    output_ports_file = 'output_ports.txt'
    output_time_intervals_file = 'output_time_intervals.txt'

    # 提取端口信息并保存到文件
    ports = extract_ports_from_pcap(file_path)
    ports_data = [f"{idx} {port}" for idx, port in enumerate(ports, start=1)]
    save_to_file(ports_data, output_ports_file)
    print(f"Ports have been saved to {output_ports_file}")

    # 提取时间戳区间映射并保存到文件
    index_mapping, min_time, max_time = extract_time_intervals_from_pcap(file_path)
    interval_size = (max_time - min_time) / index_num
    time_intervals_data = []
    for index in set(index_mapping):
        start_time = min_time + index * interval_size
        end_time = min_time + (index + 1) * interval_size
        time_intervals_data.append(f"{index} {start_time} {end_time}")
    save_to_file(time_intervals_data, output_time_intervals_file)
    print(f"Time intervals have been saved to {output_time_intervals_file}")
