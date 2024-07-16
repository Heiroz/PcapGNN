from model.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
import model.utils as utils
import pytorch_lightning as pl
import torch
from torch import nn
from model.transformer_model import GraphTransformer
from model.train_loss import TrainLoss
import torch.nn.functional as F
import time
from postprocess import construct_table

class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, input_dims, output_dims, node_types, edge_types, nodes_dist):
        super().__init__()

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.T = 500
        self.train_loss = TrainLoss(lambda_train=[5, 0])
        self.node_dist = nodes_dist
        self.model = GraphTransformer(
            n_layers=6,
            input_dims=input_dims,
            hidden_mlp_dims={'X': 256 * 8, 'E': 128 * 8, 'y': 128},
            hidden_dims={'dx': 256 * 8, 'de': 64 * 8, 'dy': 64, 'n_head': 8, 'dim_ffX': 256 * 8, 'dim_ffE': 128 * 8, 'dim_ffy': 128},
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )
        self.noise_schedule = PredefinedNoiseScheduleDiscrete('cosine', 500)

        x_marginals = node_types / torch.sum(node_types)

        e_marginals = edge_types / torch.sum(edge_types)
        
        self.transition_model = MarginalUniformTransition(
            x_marginals=x_marginals, 
            e_marginals=e_marginals,
            y_classes=self.ydim_output
        )
        self.limit_dist = utils.PlaceHolder(
            X=x_marginals, 
            E=e_marginals,
            y=torch.ones(self.ydim_output) / self.ydim_output
        )

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = 50
        self.number_chain_steps = 50

        self.last_pred = None

    def training_step(self, data, i):
        dense_data, node_mask = utils.to_dense(
            x=data.x, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr,
            batch=data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        X = X.float()
        E = E.float()
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        pred = self.forward(noisy_data, node_mask)        
        self.last_pred = pred
        loss = self.train_loss(
            masked_pred_X=pred.X, 
            masked_pred_E=pred.E, 
            pred_y=pred.y,
            true_X=X, 
            true_E=E, 
            true_y=data.y,
            log=i % self.log_every_steps == 0
        )
        return {'loss': loss}

    def forward(self, noisy_data, node_mask):
        """ Concatenates extra data to the noisy data, then calls the network. """
        X = noisy_data['X_t'].float()
        E = noisy_data['E_t'].float()
        y = noisy_data['y_t'].float()
        return self.model(X, E, y, node_mask)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0002, amsgrad=True, weight_decay=1e-12)
    
    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)

        sampled_t = utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=1024)
        E_t = F.one_hot(sampled_t.E, num_classes=1024)
        bs, n, num_attr_X, one_hot_dim_X = X_t.shape
        X_t = X_t.view(bs, n, num_attr_X * one_hot_dim_X)
        bs, _, _, num_attr_E, one_hot_dim_E = E_t.shape
        E_t = E_t.view(bs, n, n, num_attr_E * one_hot_dim_E)

        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()

    def on_train_epoch_end(self):
        if self.current_epoch != 39:
            return
        sample_graphs = []
        id = 0
        samples_left_to_generate = 100
        samples_left_to_save = 3
        chains_left_to_save = 2
        while samples_left_to_generate > 0:
            bs = 1
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            sample_graphs.extend(
                self.sample_batch(
                    id, 
                    to_generate, 
                    num_nodes=None, 
                    save_final=to_save,
                    keep_chain=chains_save, 
                    number_chain_steps=self.number_chain_steps
                )
            )
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        filename = f'generated_samples.csv'
        construct_table(sample_graphs, filename, 'port_index_mapping.txt', 'time_index_mapping.txt')


    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int, save_final: int, num_nodes=None):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        z_T = utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1), X.size(2)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2), E.size(3)))

        chain_X = torch.zeros(chain_X_size).to(self.device)
        chain_E = torch.zeros(chain_E_size).to(self.device)

        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        sampled_s = sampled_s.mask(node_mask)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain
            chain_E[0] = final_E_chain

            chain_X = utils.reverse_tensor(chain_X)
            chain_E = utils.reverse_tensor(chain_E)

            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            node_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            X_encodings = node_types.size(1) // 1024
            E_encodings = edge_types.size(2) // 1024

            if torch.all(node_types.view(n, X_encodings, 1024) == 0):
                node_types = torch.full((n, X_encodings), -1, dtype=torch.int64)
            else:
                node_types = node_types.view(n, X_encodings, 1024).argmax(dim=-1)

            if torch.all(edge_types.view(n, n, E_encodings, 1024) == 0):
                edge_types = torch.full((n, n, E_encodings), -1, dtype=torch.int64)
            else:
                edge_types = edge_types.view(n, n, E_encodings, 1024).argmax(dim=-1)

            graph = [node_types, edge_types]
            graph_list.append(graph)

        return graph_list        

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        pred = self.forward(noisy_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = utils.compute_batched_over0_posterior_distribution(
            X_t=X_t,
            Qt=Qt.X,
            Qsb=Qsb.X,
            Qtb=Qtb.X
        )

        p_s_and_t_given_0_E = utils.compute_batched_over0_posterior_distribution(
            X_t=E_t,
            Qt=Qt.E,
            Qsb=Qsb.E,
            Qtb=Qtb.E
        )
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        sampled_s = utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=1024).float()
        E_s = F.one_hot(sampled_s.E, num_classes=1024).float()
        bs, n, num_attr_X, one_hot_dim_X = X_s.shape
        X_s = X_s.view(bs, n, num_attr_X * one_hot_dim_X)
        bs, _, _, num_attr_E, one_hot_dim_E = E_s.shape
        E_s = E_s.view(bs, n, n, num_attr_E * one_hot_dim_E)

        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 1))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 1))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask).type_as(y_t)