from pytorch_lightning import Trainer
import torch
from model.diffusion_model import DenoisingDiffusion
from preprocess import classify_pcap_split, pcap2graph, edge_attr, node_attr, adjacency_matrix
from model.data_module import GraphDataModule
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
    
def main():
    pcap_file = 'caida_small.pcap'
    # 将caida数据集切分成若干块，防止由于训练数据过大导致程序崩溃
    ip_pairs_list = classify_pcap_split(pcap_file)
    data_list = []
    # 准备数据
    for ip_pairs in ip_pairs_list:
        G = pcap2graph(ip_pairs)
        X = node_attr(G)
        E, edge_index = edge_attr(G)
        adj = adjacency_matrix(G)
        adj = torch.tensor(adj, dtype=torch.float)
        edge_index, _ = dense_to_sparse(adj)
        y = torch.zeros([1, 1]).float()
        num_nodes = X.size(0) * torch.ones(1, dtype=torch.long)
        train_data = Data(
            x=X, 
            edge_index=edge_index, 
            edge_attr=E,
            y=y, 
            n_nodes=num_nodes
        )
        data_list.append(train_data)
    print("preprocess finished...")
    batch_size = 1
    datamodule = GraphDataModule(data_list, batch_size=batch_size)
    node_types = datamodule.node_types()
    edge_types = datamodule.edge_counts()
    input_dims = {'X': X.size(1), 'E': E.size(1), 'y': y.size(1)}
    output_dims = {'X': X.size(1), 'E': E.size(1), 'y': y.size(1)}
    n_nodes = datamodule.node_counts()
    nodes_dist = DistributionNodes(n_nodes)
    model = DenoisingDiffusion(input_dims, output_dims, node_types, edge_types, nodes_dist)
    trainer = Trainer(max_epochs=40, log_every_n_steps=50, precision='16-mixed')
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()