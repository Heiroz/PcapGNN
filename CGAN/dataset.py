import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class FlowDataset(Dataset):
    def __init__(self, flow_data):
        self.flow_data = flow_data

    def __len__(self):
        return len(self.flow_data)

    def __getitem__(self, idx):
        flow_info = self.flow_data[idx]
        flow_vector = flow_info['flow_vector']
        remaining_features = flow_info['remaining_features']
        return flow_vector, remaining_features


def collate_fn(batch):
    batch_flow_vectors = [item[0] for item in batch]
    batch_remaining_features = [item[1] for item in batch]
    
    batch_flow_vectors = torch.stack(batch_flow_vectors, dim=0)
    batch_remaining_features = torch.stack(batch_remaining_features, dim=0)
    
    return batch_flow_vectors, batch_remaining_features

def get_data_loader(flow_data, batch_size, world_size, rank):
    dataset = FlowDataset(flow_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=70)
    return data_loader