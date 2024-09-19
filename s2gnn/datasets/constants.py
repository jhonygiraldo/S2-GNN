from torch_geometric.datasets import (
    Planetoid, 
    Amazon, 
    Coauthor, 
    WebKB, 
    WikipediaNetwork, 
    Actor, 
    LRGBDataset, 
    TUDataset
)


# Path to the folder containing the datasets
DATA_PATH = 'data'

# Dataset names
DATASET_PATHS = {
        'cancer_b' : ['cancer_binary', 'cancer'],
        'cancer' : ['cancer', 'cancer'],
        '20news' : ['20newsgroups', '20news'],
        'activity' : ['activity_recognition', 'activity'],
        'isolet' : ['isolet', 'isolet'],
        'PascalVOC-SP' : ['pascal', 'pascal'],
        'COCO-SP' : ['coco', 'coco']
    }

# Dataset loaders
DATASET_LOADERS = {
    'Cora': Planetoid,
    'Citeseer': Planetoid,
    'Pubmed': Planetoid,
    'Cornell': WebKB,
    'Texas': WebKB,
    'Wisconsin': WebKB,
    'chameleon': WikipediaNetwork,
    'squirrel': WikipediaNetwork,
    'Actor': Actor,
    'Computers': Amazon,
    'Photo': Amazon,
    'CoauthorCS': Coauthor,
    'PascalVOC-SP': LRGBDataset,
    'COCO-SP': LRGBDataset,
}

# Special case datasets with specific batch sizes
BATCHED_DATASET_LOADERS = {
    'MUTAG': (TUDataset, 188),
    'ENZYMES': (TUDataset, 600),
    'PROTEINS': (TUDataset, 1113)
}