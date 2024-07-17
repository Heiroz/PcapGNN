import torch
from trainer import CGANTrainer
from get_flows import get_flows
from dataset import get_data_loader
from generator import Generator
from discriminator import Discriminator

def get_num_attributes(flows):
    sample_flow = flows[0]
    num_condition = sample_flow['flow_vector'].shape[0]
    num_output = sample_flow['remaining_features'].shape[1]
    return num_condition, num_output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flows, start_time, num_pkts = get_flows('caida_small.pcap')
    
    condition_dim, output_dim = get_num_attributes(flows)
    noisy_size = 1024 * 8
    num_epochs = 200

    generator = Generator(noisy_size, output_dim, condition_dim).to(device)
    discriminator = Discriminator(output_dim, condition_dim).to(device)

    # 使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    data_loader = get_data_loader(flows, batch_size=1)

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        noisy_dim=noisy_size,
        num_epochs=num_epochs,
        device=device
    )

    trainer.train()

if __name__ == "__main__":
    main()
