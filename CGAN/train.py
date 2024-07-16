from trainer import CGANTrainer
from get_flows import get_flows
from dataset import get_data_loader
from generator import Generator, Generator_sigmoid
from discriminator import Discriminator

def get_num_attributes(flows):
    sample_flow = flows[0]
    num_condition = sample_flow['flow_vector'].shape[0]
    num_output = sample_flow['remaining_features'].shape[1]
    return num_condition, num_output

def main():
    flows = get_flows('caida_small.pcap')
    
    condition_dim, output_dim = get_num_attributes(flows)
    noisy_size = 1024
    num_epochs = 200
    generator = Generator(noisy_size, output_dim, condition_dim)
    discriminator = Discriminator(output_dim, condition_dim)

    data_loader = get_data_loader(flows, batch_size=1)

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        noisy_dim=noisy_size,
        num_epochs=num_epochs
    )

    
    trainer.train()

if __name__ == "__main__":
    main()
