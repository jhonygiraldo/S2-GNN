import argparse
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args(train_mode:bool = True):
    """
    Parse command-line arguments for running experiments.

    args:
    - train_mode (bool): Flag to indicate if the arguments are for training mode. Default is True.

    returns:
    - dict: Dictionary of parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Arguments for running all the experiment.'
    )

    # Common arguments
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--graph_classification', action='store_true', default=False,
                        help='Is this experiment a graph classification problem?')
    parser.add_argument('--GraphDifussion', action='store_true', default=False,
                        help='Activate Graph Difussion preprocessing.')
    parser.add_argument('--GNN', type=str, default='SSobGNN',
                        help=('Name of graph neural network: SSobGNN, SobGNN, GCN, Cheby, '
                              'kGNN, GAT, Transformer, SGC, ClusterGCN, FiLM, SuperGAT, '
                              'GATv2, ARMA, SIGN'))
    parser.add_argument('--dataset', type=str, default='cancer_b',
                        help=('Name of dataset, options: {cancer_b, cancer, 20news, activity, isolet, '
                              'Cornell, Texas, Wisconsin, chameleon, Actor, squirrel, Cora, Citeseer, '
                              'Pubmed, COCO-S, PascalVOC-SP, MUTAG, ENZYMES, PROTEINS}'))
    parser.add_argument('--graph', type=str, default='knn',
                        help='Type of graph: knn or learned graph.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU index.')
    parser.add_argument('--alphaGDC', type=float, default=0.05,
                        help='Alpha value for graph diffusion method.')
    parser.add_argument('--k', type=int, default=128,
                        help='K value for graph diffusion method.')
    parser.add_argument('--undirected', action='store_true', default=True,
                        help='Set to not symmetrize adjacency.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show detailed results.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs.')

    # Train-specific arguments
    if train_mode:
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 regularization).')
        parser.add_argument('--hidden_units', type=int, default=16,
                            help='Number of hidden units in the model.')
        parser.add_argument('--n_layers_set', nargs="+", type=int, default=[2],
                            help='List of layer counts.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--aggregation', type=str, default='linear',
                            help='Aggregation type for S2-GNN: linear, concat, attention.')
        parser.add_argument('--k_cheby', type=int, default=1,
                            help='K value for ChebConv.')
        parser.add_argument('--k_sign', type=int, default=2,
                            help='K value for SIGN.')
        parser.add_argument('--alpha', type=int, default=6,
                            help='Alpha parameter for S-SobGNN.')
        parser.add_argument('--heads_attention', type=int, default=1,
                            help=('Number of attention heads for GAT, Transformer, '
                                  'SuperGAT, and GATv2.'))
        parser.add_argument('--epsilon', type=float, default=1.5,
                            help='Epsilon parameter for S-SobGNN or SobGNN.')
        parser.add_argument('--hyperparam_mode', action='store_true', default=False,
                            help='Activate hyperparameter tuning mode.')
    else:
        # Hyperparameter tuning mode
        parser.add_argument('--hyperparam_mode', action='store_true', default=True,
                            help='Activate hyperparameter tuning mode.')
        parser.add_argument('--iterations', type=int, default=100,
                            help='Number of iterations for random search.')

    args = parser.parse_args()
    logger.info(f"args = {args}")

    return vars(args)  # Returns arguments as a dictionary