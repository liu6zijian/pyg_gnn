# -*- coding: utf-8 -*-
"""
@Time   : 2020/5/28

@Author : Shen Fang
"""
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


# create a toy dataset
def toy_dataset(num_nodes, num_node_features, num_edges):
    x = np.random.randn(num_nodes, num_node_features)  # node features
    edge_index = np.random.randint(low=0, high=num_nodes-1, size=[2, num_edges], dtype=np.long)  # [2, num_edges]

    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))

    return data


# In Memory Dataset
class PyGToyDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        super(PyGToyDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):
        return ["origin_dataset"]

    @property
    def processed_file_names(self):
        return ["toy_dataset.pt"]

    def download(self):
        pass

    def process(self):
        # 100 samples, each sample is a graph with 32 nodes and 42 edges, each node has a feature dimension of 3.
        data_list = [toy_dataset(num_nodes=32, num_node_features=3, num_edges=42) for _ in range(100)]
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_file_names[0])


if __name__ == '__main__':
    # toy_sample = toy_dataset(num_nodes=32, num_node_features=3, num_edges=42)
    # print(toy_sample)
    toy_data = PyGToyDataset(save_root="toy")  # 100 samples, each sample is a graph
    # print(toy_data[0])
    data_loader = DataLoader(toy_data, batch_size=5, shuffle=True)

    for batch in data_loader:
        print(batch)
