from pathlib import Path
from typing import Dict, Any
import yaml
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args: config_path (str): The path to the configuration file.

    Returns: Dict[str, Any]: A dictionary containing the parsed configuration.

    Raises: FileNotFoundError: If the configuration file does not exist.
            KeyError: If required configuration fields are missing.
    """
    config_file = Path(config_path)
    
    # Check if the configuration file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_file.absolute()}")

    # Load the YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required configuration fields
    required_keys = [
        'train_dir', 
        'valid_dir',
        'test_dir'
    ]
    
    for key in required_keys:
        if not _nested_key_exists(config, key):
            raise KeyError(f"Missing required configuration field: {key}")

    return config


def _nested_key_exists(config: Dict, dotted_key: str) -> bool:
    """
    Checks if a nested key exists in the configuration dictionary.

    Args:
        config (Dict): The configuration dictionary.
        dotted_key (str): A string representing the dotted path to the key (e.g., 'data.earthquake.train_dir').

    Returns:
        bool: True if the nested key exists, False otherwise.
    """
    keys = dotted_key.split('.')
    current = config
    for k in keys:
        if k not in current:
            return False
        current = current[k]
    return True


def load_raw_data(config_path, dataset_name) -> Dict:
    """
    Loads the raw dataset specified in the configuration file.

    Args:
        dataset_name (str): The name of the dataset. Corresponds to a key in the config.yaml file under 'data'.

    Returns:
        Dict: A dictionary containing 'train', 'dev', and 'test'.

    Raises:
        FileNotFoundError: If any of the dataset files do not exist.
    """
    config = load_config(config_path)
    
    # Retrieve dataset configuration from the loaded config
    try:
        dataset_cfg = config
        paths = {
            'train': Path(dataset_cfg['train_dir']),
            'dev': Path(dataset_cfg['valid_dir']),  # 'valid_dir' corresponds to 'dev.pkl'
            'test': Path(dataset_cfg['test_dir'])
        }
    except KeyError as e:
        raise KeyError(f"Invalid dataset configuration item: {e}")

    # Check if the dataset files exist
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{split} dataset file does not exist: {path.absolute()}")

    # Return the loaded data as pandas DataFrames
    return {
        'train': pd.read_pickle(paths['train']),
        'dev': pd.read_pickle(paths['dev']),
        'test': pd.read_pickle(paths['test'])
    }