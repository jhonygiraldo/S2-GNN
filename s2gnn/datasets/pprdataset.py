import pickle

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_dense_adj

from s2gnn.datasets.constants import DATA_PATH
from s2gnn.datasets.helpers import get_dataset


class PPRDataset(InMemoryDataset):
    """
    Dataset preprocessed with Graph Diffusion Convolution (GDC) using Personalized 
    PageRank (PPR) diffusion. Note that this implementation is not scalable as it 
    directly inverts the adjacency matrix.
    """

    def __init__(
            self, 
            name: str = 'Cora', 
            use_lcc: bool = True, 
            alpha: float = 0.1, 
            k: int = 16, 
            eps: float = None
        ):
        """
        Initialize the PPRDataset.

        args:
        - name (str): The name of the dataset.
        - use_lcc (bool): Whether to use the largest connected component.
        - alpha (float): Damping factor for PPR computation (0 < alpha < 1).
        - k (int): Number of top entries to keep per column.
        - eps (float, optional): Threshold below which elements are clipped.

        returns:
        - None
        """
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps

        super(PPRDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_filenames(self) -> list:
        """
        Get the list of raw filenames.

        returns:
        - list: An empty list as there are no raw files.
        """
        return []

    @property
    def processed_filenames(self) -> list:
        """
        Get the list of processed filenames.

        returns:
        - list: A list containing the processed filename.
        """
        return [f'{self}.pt']

    def download(self):
        """
        Download the dataset. This method is not implemented as the dataset is assumed to be pre-downloaded.

        returns:
        - None
        """
        pass

    @staticmethod
    def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
        """
        Convert the edge index of a dataset into an adjacency matrix.

        args:
        - dataset (InMemoryDataset): PyTorch Geometric InMemoryDataset object.

        returns:
        - np.ndarray: A numpy array representing the adjacency matrix.
        """
        num_nodes = dataset.data.x.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        edge_index = dataset.data.edge_index.numpy()
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        return adj_matrix

    @staticmethod
    def get_ppr_matrix(adj_matrix: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Compute the Personalized PageRank (PPR) matrix from an adjacency matrix.

        args:
        - adj_matrix (np.ndarray): The adjacency matrix of the graph.
        - alpha (float): Damping factor for PPR computation (0 < alpha < 1).

        returns:
        - np.ndarray: The Personalized PageRank matrix.
        """
        identity_matrix = np.eye(num_nodes)
        num_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix + identity_matrix
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
        H = D_tilde @ A_tilde @ D_tilde

        return alpha * np.linalg.inv(identity_matrix - (1 - alpha) * H)

    @staticmethod
    def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
        """
        Keep only the top-k entries per column of the matrix.

        args:
        - A (np.ndarray): The adjacency matrix.
        - k (int): Number of top entries to keep per column.

        returns:
        - np.ndarray: The matrix with only top-k entries per column.
        """
        num_nodes = A.shape[0]
        top_k_indices = np.argsort(A, axis=0)[-k:]
        mask = np.zeros_like(A, dtype=bool)
        row_idx = np.arange(num_nodes)
        mask[top_k_indices, row_idx] = True
        A = np.where(mask, A, 0.0)
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1  # Avoid division by zero

        return A / norm

    @staticmethod
    def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
        """
        Zero out elements in the matrix below a certain threshold and normalize.

        args:
        - A (np.ndarray): The adjacency matrix.
        - eps (float): The threshold below which elements are clipped.

        returns:
        - np.ndarray: The clipped and normalized matrix.
        """
        A_clipped = np.where(A >= eps, A, 0.0)
        norm = A_clipped.sum(axis=0)
        norm[norm <= 0] = 1  # Avoid division by zero

        return A_clipped / norm

    def process(self):
        """
        Process the dataset to compute the PPR matrix and create the Data object.

        returns:
        - None
        """
        base = self._load_base_dataset()
        adj_matrix = self._get_adjacency_matrix(base)
        ppr_matrix = self.get_ppr_matrix(adj_matrix, alpha=self.alpha)

        if self.k:
            ppr_matrix = self.get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            ppr_matrix = self.get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError("Either 'k' or 'eps' must be provided.")

        data = self._create_data_object(base, ppr_matrix)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def _load_base_dataset(self) -> InMemoryDataset:
        """
        Load the base dataset depending on the dataset name.

        returns:
        - InMemoryDataset: The loaded base dataset.
        """
        if self.name in ['chameleon', 'squirrel', 'Actor']:
            filename = f'data/{self.name}.pkl'
            with open(filename, 'rb') as f:
                base = pickle.load(f)[0]
        else:
            base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        return base

    def _get_adjacency_matrix(self, base: InMemoryDataset) -> np.ndarray:
        """
        Generate the adjacency matrix from the base dataset.

        args:
        - base (InMemoryDataset): The base dataset.

        returns:
        - np.ndarray: The adjacency matrix.
        """
        if self.name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
            adj_matrix = to_dense_adj(base.edge_index).squeeze(0).numpy()
        else:
            adj_matrix = self.get_adj_matrix(base)
        return adj_matrix

    def _create_data_object(self, base: InMemoryDataset, ppr_matrix: np.ndarray) -> Data:
        """
        Create a PyG Data object from the base dataset and PPR matrix.

        args:
        - base (InMemoryDataset): The base dataset.
        - ppr_matrix (np.ndarray): The Personalized PageRank matrix.

        returns:
        - Data: The created PyG Data object.
        """
        edges_i, edges_j, edge_attr = [], [], []
        for i, row in enumerate(ppr_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(row[j])

        edge_index = torch.LongTensor([edges_i, edges_j])
        edge_attr = torch.FloatTensor(edge_attr)

        if self.name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
            data = Data(
                x=base.x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=base.y,
                train_mask=torch.zeros(base.y.size(0), dtype=torch.bool),
                test_mask=torch.zeros(base.y.size(0), dtype=torch.bool),
                val_mask=torch.zeros(base.y.size(0), dtype=torch.bool),
                batch=base.batch
            )
        else:
            data = Data(
                x=base.data.x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=base.data.y,
                train_mask=torch.zeros(base.data.train_mask.size(0), dtype=torch.bool),
                test_mask=torch.zeros(base.data.test_mask.size(0), dtype=torch.bool),
                val_mask=torch.zeros(base.data.val_mask.size(0), dtype=torch.bool)
            )
        return data

    def __str__(self) -> str:
        """
        Get the string representation of the dataset.

        returns:
        - str: The string representation of the dataset.
        """
        return (
            f'{self.name}_ppr_alpha={self.alpha}_k={self.k}_'
            f'eps={self.eps}_lcc={self.use_lcc}'
        )