__author__ = "Stefan Weißenberger and Johannes Klicpera, modified by Jhony H. Giraldo"
__license__ = "MIT"

import os
import logging
from typing import List, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from s2gnn.datasets.constants import (
    DATA_PATH, 
    DATASET_LOADERS,
    BATCHED_DATASET_LOADERS
)
from s2gnn.datasets.seeds import DEVELOPMENT_SEED

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_special_case_dataset(name: str, path: str) -> InMemoryDataset:
    """
    Load a special case dataset that requires batching.

    args:
    - name (str): The name of the dataset.
    - path (str): The path to the dataset.

    returns:
    - InMemoryDataset: The loaded dataset.
    """
    dataset_class, batch_size = BATCHED_DATASET_LOADERS[name]
    dataset = dataset_class(path, name)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loaded_dataset = next(iter(dataloader))
    loaded_dataset.slices = dataset.slices

    return loaded_dataset


def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    """
    Get a dataset by name and optionally process it to use the largest connected component.

    args:
    - name (str): The name of the dataset.
    - use_lcc (bool): Whether to use the largest connected component (default is True).

    returns:
    - InMemoryDataset: The processed dataset.
    """
    path = os.path.join(DATA_PATH, name)

    if name in BATCHED_DATASET_LOADERS:
        dataset = load_special_case_dataset(name, path)
    elif name in DATASET_LOADERS:
        loader_class = DATASET_LOADERS[name]
        dataset = loader_class(path, name)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if use_lcc and name not in BATCHED_DATASET_LOADERS:
        lcc = get_largest_connected_component(dataset)
        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    return dataset


def visualize_matrix(
        matrix: np.ndarray, 
        cmap: str = 'viridis', 
        title: str = 'Matrix Visualization', 
        output_file: str = None
    ) -> None:
    """
    Visualize a matrix with colors and optionally save it as a PDF.

    args:
    - matrix (np.ndarray): The matrix to visualize.
    - cmap (str): The colormap to use (default is 'viridis').
    - title (str): The title of the plot.
    - output_file (str, optional): The file path to save the plot as a PDF (default is None).

    returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    if output_file:
        plt.savefig(output_file, format='pdf')
        logger.info(f"Plot saved as {output_file}")
    else:
        plt.show()


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    """
    Find the connected component of a graph starting from a specified node.

    args:
    - dataset (InMemoryDataset): The dataset containing the graph.
    - start (int): The starting node index (default is 0).

    returns:
    - set: A set of nodes that are in the same connected component as the start node.
    """
    visited_nodes = set()
    queued_nodes = {start}
    row, col = dataset.data.edge_index.numpy()

    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.add(current_node)

        # Find neighbors of the current node
        neighbors = col[np.where(row == current_node)[0]]

        # Add unvisited neighbors to the queue
        queued_nodes.update(n for n in neighbors if n not in visited_nodes)

    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    """
    Get the largest connected component from the dataset.

    args:
    - dataset (InMemoryDataset): The dataset containing graph data.

    returns:
    - np.ndarray: An array of node indices in the largest connected component.
    """
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    largest_component = set()

    while remaining_nodes:
        start = remaining_nodes.pop()  # Get and remove a node from remaining_nodes
        component = get_component(dataset, start)
        
        if len(component) > len(largest_component):
            largest_component = component
        
        remaining_nodes.difference_update(component)  # Remove the nodes in the component from remaining_nodes

    return np.array(list(largest_component))


def get_node_mapper(lcc: np.ndarray) -> Dict[int, int]:
    """
    Create a mapping from old node indices to new indices.

    args:
    - lcc (np.ndarray): An array of node indices in the largest connected component.

    returns:
    - Dict[int, int]: A dictionary mapping old node indices to new indices.
    """
    return {node: idx for idx, node in enumerate(lcc)}


def remap_edges(
        edges: List[Tuple[int, int]], 
        mapper: Dict[int, int]
    ) -> List[List[int]]:
    """
    Remap the edges using the node mapper.

    args:
    - edges (List[Tuple[int, int]]): A list of tuples where each tuple represents an edge.
    - mapper (Dict[int, int]): A dictionary mapping old node indices to new indices.

    returns:
    - List[List[int]]: A list of two lists: row indices and column indices of the remapped edges.
    """
    row, col = zip(*edges)
    row_mapped = [mapper[node] for node in row]
    col_mapped = [mapper[node] for node in col]

    return [row_mapped, col_mapped]


def set_train_val_test_split(
        seed: int,
        data: Data,
        dataset_name: str = 'Cora',
        num_development: int = 1500,
        num_per_class: int = 20
    ) -> Data:
    """
    Set the train, validation, and test splits for the dataset.

    args:
    - seed (int): Random seed for reproducibility.
    - data (Data): PyTorch Geometric Data object containing node features and labels.
    - dataset_name (str): Name of the dataset (e.g., 'Cora').
    - num_development (int): Number of nodes to use for the development set.
    - num_per_class (int): Number of nodes per class to use for training in class-specific datasets.

    returns:
    - Data: The input Data object with updated masks.
    """
    num_nodes = data.y.shape[0]
    rnd_state = np.random.RandomState(DEVELOPMENT_SEED)

    # Split nodes into development and test sets
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = np.setdiff1d(np.arange(num_nodes), development_idx)

    # Create random state for reproducibility
    rnd_state = np.random.RandomState(seed)
    # Initialize train indices
    train_idx = []

    # Set train and validation splits
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        for c in range(data.y.max() + 1):
            class_idx = development_idx[data.y[development_idx].cpu() == c]
            if len(class_idx) > num_per_class:
                train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
            else:
                train_idx.extend(class_idx)
    else:
        num_train = int(0.1 * num_nodes)
        train_idx = rnd_state.choice(development_idx, num_train, replace=False)

    val_idx = np.setdiff1d(development_idx, train_idx)

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data