import os
import pickle
import logging 
from typing import Dict, Tuple, Any

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected, 
    remove_self_loops, 
    to_dense_adj
)
from scipy.io import loadmat

from s2gnn.nets.models import (
    SSobGNN, 
    GCN, 
    Cheby, 
    kGNN, 
    GAT,
    Transformer, 
    SGC, 
    ClusterGCN, 
    FiLM,
    SuperGAT, 
    GATv2, 
    ARMA, 
    SIGN
)
from s2gnn.datasets.pprdataset import PPRDataset
from s2gnn.datasets.helpers import get_dataset


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_results(
        config: Dict[str, Any], 
        results: Dict[str, Any], 
        n_layers: int
    ) -> None:
    """
    Save the results of the model training or hyperparameter tuning.

    args:
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.
    - results (Dict[str, Any]): Dictionary containing the results to be saved.
    - n_layers (int): Number of layers in the model.

    returns:
    - None
    """
    # Generate the base file name
    file_name_base = get_filename_base(n_layers, config)
    
    # Determine the output directory based on the hyperparameter tuning mode
    sub_folder = (f'{config["GNN"]}_hyperTuning' 
                  if config["hyperparam_mode"] 
                  else config["GNN"])
    path = os.path.join(config["output_folder"], sub_folder, config["graph"])
    
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Determine the result type based on the mode and save the results
    result_type = 'hyper' if config["hyperparam_mode"] else 'train'
    with open(os.path.join(path, f'{file_name_base}.pkl'), 'wb') as f:
        pickle.dump(results[result_type], f)


def get_filename_base(n_layers: int, config: Dict[str, Any]) -> str:
    """
    Generate the base filename for saving results based on the configuration.

    args:
    - n_layers (int): Number of layers in the model.
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - str: The base filename.
    """
    # Base filename components
    filename_base = (
        f'{config["dataset"]}_nL_{n_layers}_hU_{config["hidden_units"]}_lr_{config["lr"]}_'
        f'wD_{config["weight_decay"]}_dr_{config["dropout"]}_mE_{config["epochs"]}'
    )
    
    # Additional filename components based on the GNN type
    gnn_specifics = {
        'SSobGNN': f'_alpha_{config["alpha"]}_epsilon_{config["epsilon"]}_aggregation_{config["aggregation"]}',
        'SobGNN': f'_alpha_{config["alpha"]}_epsilon_{config["epsilon"]}_aggregation_{config["aggregation"]}',
        'Cheby': f'_KCheby_{config["k_cheby"]}',
        'GAT': f'_heads_attention_{config["heads_attention"]}',
        'Transformer': f'_heads_attention_{config["heads_attention"]}',
        'SuperGAT': f'_heads_attention_{config["heads_attention"]}',
        'GATv2': f'_heads_attention_{config["heads_attention"]}',
        'SIGN': f'_KSIGN_{config["k_sign"]}'
    }
    
    # Append GNN-specific components if applicable
    complement_filename = gnn_specifics.get(config["GNN"], "")
    filename_base += complement_filename

    return filename_base


def get_graph_data_graph_classification(
        data_tmp: Data, 
        config: Dict[str, Any]
    ) -> Data:
    """
    Prepare graph data for graph classification tasks.

    args:
    - data_tmp (Data): Temporary data object containing the graph data.
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - Data: The prepared graph data for classification.
    """
    # Default edge_index and edge_attr
    edge_index = data_tmp.edge_index
    edge_attr = data_tmp.edge_attr

    if config["GNN"] == 'SSobGNN':
        adj = to_dense_adj(data_tmp.edge_index).squeeze(0)
        adj[data_tmp.edge_index[0], data_tmp.edge_index[1]] = data_tmp.edge_attr
        adj_tilde = adj + config["epsilon"] * torch.eye(adj.shape[0])

        edge_index_list = []
        edge_attr_list = []

        for rho in range(1, config["alpha"] + 1):
            sparse_sobolev_term = torch.pow(adj_tilde, rho)
            edge_index_temp = sparse_sobolev_term.nonzero().T
            edge_index_list.append(edge_index_temp)
            edge_attr_list.append(
                sparse_sobolev_term[edge_index_temp[0, :],
                                    edge_index_temp[1, :]]
            )

        # Concatenate all rho terms to get the final edge_index and edge_attr
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list)

    # Create a new Data object with the appropriate attributes
    data = Data(
        x=data_tmp.x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data_tmp.y,
        batch=data_tmp.batch,
        num_features=data_tmp.x.shape[1],
        num_classes=data_tmp.num_classes
    )

    return data


def get_graph_data(
        points: torch.Tensor, 
        labels: torch.Tensor, 
        num_classes: int, 
        adj: torch.Tensor, 
        config: Dict[str, Any]
    ) -> Data:
    """
    Prepare graph data for various GNN models.

    args:
    - points (torch.Tensor): Node features.
    - labels (torch.Tensor): Node labels.
    - num_classes (int): Number of classes.
    - adj (torch.Tensor): Adjacency matrix.
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - Data: The prepared graph data.
    """

    data = None

    if config["GNN"] in ['SSobGNN', 'SobGNN']:
        adj_tilde = adj + config["epsilon"] * torch.eye(adj.shape[0])
        edge_index = []
        edge_attr = []

        for rho in range(1, config["alpha"] + 1):
            sobolev_term = (torch.pow(adj_tilde, rho) 
                            if config["GNN"] == 'SSobGNN' 
                            else torch.matrix_power(adj_tilde, rho))
            edge_index_temp = sobolev_term.nonzero().T
            edge_index.append(edge_index_temp)
            edge_attr.append(
                sobolev_term[edge_index_temp[0, :], 
                             edge_index_temp[1, :]]
            )

        data = Data(
            x=points, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=labels
        )

    elif config["GNN"] == 'SIGN':
        xs = []
        tensor_ones = torch.ones(adj.shape[0], 1)
        diag_elements = torch.mm(adj, tensor_ones)
        diag_elements = torch.div(tensor_ones, torch.sqrt(diag_elements))
        diag_tilde = torch.diag(diag_elements.view(-1))
        filtering_function = torch.spmm(
                                diag_tilde, 
                                torch.spmm(adj, diag_tilde)
                            )

        for k_sign in range(1, config["k_sign"] + 1):
            higher_term = torch.matrix_power(filtering_function, k_sign)
            xs.append(torch.mm(higher_term, points))

        data = Data(x=points, y=labels)
        data.xs = xs

    elif config["GNN"] in ['GCN', 'Cheby', 'kGNN', 'GAT', 'Transformer', 'SGC',
                           'ClusterGCN', 'FiLM', 'SuperGAT', 'GATv2', 'ARMA']:
        edge_index = adj.nonzero().T
        edge_attr = adj[edge_index[0, :], edge_index[1, :]]
        data = Data(
            x=points, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=labels
        )

    if data:
        data.num_features = points.shape[1]
        data.num_classes = num_classes

    return data


def load_graph_data(
        config: Dict[str, Any], 
        root_folder: str = 'data', 
        name_folder: str = None, 
        init_name: str = None
    ) -> Tuple[Data, int]:
    """
    Load graph data from files or datasets based on the configuration.

    args:
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.
    - root_folder (str, optional): Path to the root folder containing the data. Default is 'data'.
    - name_folder (str, optional): Name of the folder containing the data. Default is None.
    - init_name (str, optional): Initial name of the data file. Default is None.

    returns:
    - Tuple[Data, int]: The loaded graph data and the number of nodes.
    """
    if name_folder:
        logger.info(f'Loading {config["graph"]} graph')
        matfile = loadmat(
                    os.path.join(
                        root_folder, 
                        name_folder, 
                        f'{init_name}_graph_{config["graph"]}.mat'
                    )
                )

        # Extract data from matfile
        adj_matrices = matfile['adj_matrices']
        points = matfile['points']
        labels = matfile['label_bin']
        num_classes = labels.shape[1]
        labels = torch.LongTensor(np.where(labels)[1])
        adj = torch.FloatTensor(adj_matrices[0][0])
        points = torch.FloatTensor(points[0][0])

        data = get_graph_data(points, labels, num_classes, adj, config)

        return data, adj.shape[0]

    if config["dataset"] in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        config["graph_classification"] = True
        config["k"] = 15
        dataset = PPRDataset(
            name=config["dataset"],
            use_lcc=False,
            alpha=config["alphaGDC"],
            k=config["k"]
        )
        dataset.data.num_classes = torch.unique(dataset.data.y).shape[0]
        data = get_graph_data_graph_classification(dataset.data, config)

        return data, data.y.shape[0]

    if config["GraphDifussion"]:
        dataset = PPRDataset(
            name=config["dataset"],
            use_lcc=True,
            alpha=config["alphaGDC"],
            k=config["k"]
        )

        if config["undirected"]:
            dataset._data.edge_index, \
            dataset._data.edge_attr = to_undirected(
                                            dataset._data.edge_index,
                                            dataset._data.edge_attr
                                        )
        dataset._data.edge_index, \
        dataset._data.edge_attr = remove_self_loops(
                                            dataset._data.edge_index,
                                            dataset._data.edge_attr
                                        )
    else:
        if config["dataset"] in ['chameleon', 'squirrel', 'Actor']:
            file_name = os.path.join(root_folder, f'{config["dataset"]}.pkl')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
            dataset = dataset[0]
        else:
            dataset = get_dataset(name=config["dataset"], use_lcc=True)

        if config["undirected"]:
            dataset._data.edge_index = to_undirected(dataset._data.edge_index)

        dataset._data.edge_index = remove_self_loops(dataset._data.edge_index)[0]

    data = dataset._data
    data.num_classes = torch.unique(data.y).shape[0]

    return data, data.num_nodes


def get_model(
        config: Dict[str, Any], 
        data: Data, n_layers: int
    ) -> torch.nn.Module:
    """
    Get the GNN model based on the configuration.

    args:
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.
    - data (Data): PyTorch Geometric Data object containing node features and labels.
    - n_layers (int): Number of layers in the model.

    returns:
    - torch.nn.Module: The initialized GNN model.
    """
    # Define a dictionary that maps GNN names to their corresponding classes
    gnn_models = {
        'SSobGNN': SSobGNN,
        'SobGNN': SSobGNN,
        'GCN': GCN,
        'Cheby': Cheby,
        'kGNN': kGNN,
        'GAT': GAT,
        'Transformer': Transformer,
        'SGC': SGC,
        'ClusterGCN': ClusterGCN,
        'FiLM': FiLM,
        'SuperGAT': SuperGAT,
        'GATv2': GATv2,
        'ARMA': ARMA,
        'SIGN': SIGN
    }
    
    # Get the model class based on the GNN type in config
    model_class = gnn_models.get(config["GNN"])

    # Check if the model class exists
    if model_class:
        # Initialize the model with the corresponding parameters
        model = model_class(
            in_channels=data.num_features,
            out_channels=data.num_classes,
            number_layers=n_layers,
            kwargs=config
        )
    else:
        raise ValueError(f"Unknown GNN type: {config['GNN']}")

    return model


def set_device(config: Dict[str, Any]) -> torch.device:
    """
    Set up the computation device (CPU or GPU) based on the configuration.

    args:
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - torch.device: The device to be used for computation.
    """
    if not config.get("no_cuda", True):
        gpu_number = config.get("gpu_number", 0)
        if torch.cuda.device_count() > gpu_number:
            logger.info(f'Setting up GPU {gpu_number}...')
            config['device'] = torch.device('cuda', gpu_number)
        else:
            logger.info(f'GPU {gpu_number} not available, setting up GPU 0...')
            config['device'] = torch.device('cuda', 0)
    else:
        logger.info('GPU not available, setting up CPU...')
        config['device'] = torch.device('cpu')


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    args:
    - seed (int): The seed value to set.

    returns:
    - None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)