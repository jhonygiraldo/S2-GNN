import logging 
from typing import Dict, Any

from s2gnn.utils.tools import load_graph_data, set_device
from s2gnn.utils.parser import parse_args
from s2gnn.datasets.seeds import TEST_SEEDS
from s2gnn.datasets.constants import DATASET_PATHS
from s2gnn.runner import run


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(
        config: Dict[str, Any], 
        data_folder: str = 'data', 
        output_folder: str = 'output'
    ) -> None:
    """
    Main function to run the single training process.

    args:
    - config (dict, optional): Configuration dictionary containing dataset and hyperparameters. Default is None.
    - data_folder (str, optional): Path to the folder containing the data. Default is 'data'.
    - output_folder (str, optional): Path to the folder to save the output. Default is 'output'.

    returns:
    - None
    """

    set_device(config)

    name_folder, init_name = DATASET_PATHS.get(config["dataset"], [None, None])
    data, n = load_graph_data(config, data_folder, name_folder, init_name)

    config['output_folder'] = output_folder

    run(data, n, config, seeds=TEST_SEEDS)


if __name__ == '__main__':
    config = parse_args(train_mode=True)
    main(config, data_folder='data', output_folder='single_training')