import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')

    flows, start_time, num_pkts = get_flows('caida_small.pcap')
    
    condition_dim, output_dim = get_num_attributes(flows)
    noisy_size = 1024 * 8
    num_epochs = 200

    generator = Generator(noisy_size, output_dim, condition_dim).to(device)
    discriminator = Discriminator(output_dim, condition_dim).to(device)

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    data_loader = get_data_loader(flows, batch_size=64, world_size=world_size, rank=rank)

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        noisy_dim=noisy_size,
        num_epochs=num_epochs,
        device=device
    )

    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 6
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
