import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np

EMB_SIZE = 128
ITEM_NFEATS = 2
EDGE_NFEATS = 1
COLUMN_NFEATS = 4
ACTIVATION_FUNCTION = torch.nn.LeakyReLU()

class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
        


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__('mean')
        emb_size = EMB_SIZE
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=True)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=True)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )
        
        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
        # return self.output_module(torch.cat([output, right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        # output = self.feature_module_final(self.feature_module_left(node_features_i)  + self.feature_module_edge(edge_features) + self.feature_module_right(node_features_j))
        output = self.feature_module_final(self.feature_module_left(node_features_i) + torch.sigmoid(self.feature_module_edge(
            edge_features)) * self.feature_module_right(node_features_j))
        return output

class Actor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        emb_size = EMB_SIZE
        item_nfeats = ITEM_NFEATS
        edge_nfeats = EDGE_NFEATS
        column_nfeats = COLUMN_NFEATS

        self.item_embedding = torch.nn.Sequential(
            torch.nn.Linear(item_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.column_embedding = torch.nn.Sequential(
            torch.nn.Linear(column_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.conv_item_to_column_1 = BipartiteGraphConvolution()
        self.conv_column_to_item_1 = BipartiteGraphConvolution()
        self.conv_item_to_column_2 = BipartiteGraphConvolution()
        self.conv_column_to_item_2 = BipartiteGraphConvolution()
        self.conv_item_to_column_3 = BipartiteGraphConvolution()
        self.conv_column_to_item_3 = BipartiteGraphConvolution()

        self.column_between_gcn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )
        self.item_between_gcn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Linear(emb_size, 1, bias=True),
        )

        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, item_features, edge_indices, edge_features, column_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # for parameters in self.item_embedding.parameters():  # 打印出参数矩阵及值
            # print(parameters)
        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        column_features = self.column_embedding(column_features)

        item_features = self.conv_column_to_item_1(column_features, reversed_edge_indices, edge_features, item_features)
        column_features = self.conv_item_to_column_1(item_features, edge_indices, edge_features, column_features)
        '''
        item_features = self.conv_column_to_item_2(column_features, reversed_edge_indices, edge_features, item_features)
        column_features = self.conv_item_to_column_2(item_features, edge_indices, edge_features, column_features)
        item_features = self.conv_column_to_item_3(column_features, reversed_edge_indices, edge_features, item_features)
        column_features = self.conv_item_to_column_3(item_features, edge_indices, edge_features, column_features)
        '''

        output = self.output_module(column_features).squeeze(-1)
        return output


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = EMB_SIZE
        item_nfeats = ITEM_NFEATS
        edge_nfeats = EDGE_NFEATS
        column_nfeats = COLUMN_NFEATS

        self.item_embedding = torch.nn.Sequential(
            torch.nn.Linear(item_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.column_embedding = torch.nn.Sequential(
            torch.nn.Linear(column_nfeats, emb_size),
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION
        )

        self.conv_item_to_column_1 = BipartiteGraphConvolution()
        self.conv_column_to_item_1 = BipartiteGraphConvolution()
        self.conv_item_to_column_2 = BipartiteGraphConvolution()
        self.conv_column_to_item_2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            ACTIVATION_FUNCTION,
            torch.nn.Linear(emb_size, 1, bias=True),
        )

        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, item_features, edge_indices, edge_features, column_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        column_features = self.column_embedding(column_features)

        column_features = self.conv_item_to_column_1(item_features, edge_indices, edge_features, column_features)
        item_features = self.conv_column_to_item_1(column_features, reversed_edge_indices, edge_features,item_features)
        column_features = self.conv_item_to_column_2(item_features, edge_indices, edge_features, column_features)
        item_features = self.conv_column_to_item_2(column_features, reversed_edge_indices, edge_features, item_features)

        output = self.output_module(item_features).squeeze(-1)
        return output