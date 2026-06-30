"""Configuration file for BERT4Rec"""

# Model parameters
MODEL_CONFIG = {
    'ml-1m': {
        'max_len': 50,
        'num_blocks': 2,
        'num_heads': 2,
        'hidden_units': 256,
        'dropout': 0.2
    },
    'ml-20m': {
        'max_len': 50,
        'num_blocks': 2,
        'num_heads': 2,
        'hidden_units': 256,
        'dropout': 0.2
    },
    'beauty': {
        'max_len': 50,
        'num_blocks': 2,
        'num_heads': 2,
        'hidden_units': 256,
        'dropout': 0.2
    },
    'steam': {
        'max_len': 50,
        'num_blocks': 2,
        'num_heads': 2,
        'hidden_units': 256,
        'dropout': 0.2
    }
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 0.01,
    'epochs': 200,
    'eval_every': 10,
    'mask_prob': 0.15,
    'top_k': 10
}

# Data parameters
DATA_CONFIG = {
    'test_ratio': 0.2,
    'min_sequence_length': 5,
    'max_sequence_length': 200
}

# Evaluation parameters
EVAL_CONFIG = {
    'k_list': [1, 5, 10, 20],
    'num_workers': 4
}

def get_config(dataset_name):
    """Get configuration for specific dataset"""
    config = {
        'model': MODEL_CONFIG.get(dataset_name, MODEL_CONFIG['ml-1m']),
        'train': TRAIN_CONFIG,
        'data': DATA_CONFIG,
        'eval': EVAL_CONFIG
    }
    return config