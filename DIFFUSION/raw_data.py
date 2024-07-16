import csv
from preprocess import get_flows
def write_flows_to_csv(flow_analysis, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['src_ip', 'dst_ip', 'protocol', 'num_pkt'])  # 写入 CSV 表头

        for flow_info in flow_analysis:
            # 提取 src_ip 和 dst_ip
            src_ip = '.'.join(map(str, flow_info['flow_vector'][:4]))
            dst_ip = '.'.join(map(str, flow_info['flow_vector'][4:8]))

            # 提取 protocol 和 num_pkt
            protocol = flow_info['flow_vector'][10]  # 根据 flow_vector 的索引获取 protocol
            num_pkt = flow_info['flow_vector'][12]  # 根据 flow_vector 的索引获取 num_pkt

            # 写入一行数据到 CSV 文件
            writer.writerow([src_ip, dst_ip, protocol, num_pkt])

# 使用示例
filename = 'caida_small.pcap'
flow_analysis = get_flows(filename)  # 假设 flows 是你的原始数据
write_flows_to_csv(flow_analysis, 'flows_data.csv')
