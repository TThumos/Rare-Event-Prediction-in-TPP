from typing import Dict, List
import numpy as np

from .data_loader import load_raw_data


def extract_sequences(data, dim_process) -> Dict[str, List[List[float]]]:
    """
    Extracts time series data from the raw dataframe.

    Returns:
        Dict[str, List[List[float]]]: A dictionary containing three types of sequences: 
            'time_seqs', 'type_seqs', and 'time_delta_seqs'.
    """
    sequences = {'time_seqs': [[x["time_since_start"] for x in seq] for seq in data],
                 'type_seqs': [[x["type_event"] for x in seq] for seq in data],
                 'time_delta_seqs': [[x["time_since_last_event"] for x in seq] for seq in data]}
    
    return sequences


def create_event_buckets(data, dim_process) -> List[List[np.ndarray]]:
    """
    Creates event buckets based on event types from the raw data.

    Args: dim_process (int): The number of event types (dimension of the process).

    Returns:
        List[List[np.ndarray]]: A list of event buckets, with each bucket containing events by type.
    """
    # Initialize buckets: one list per sequence, each containing 'dim_process' empty lists
    buckets = [[[] for _ in range(dim_process)] for _ in range(len(data))]
    
    # Loop through each sequence in the data
    for i, seq in enumerate(data):
        for event in seq:
            type_event = event['type_event']
            # Ensure event type is within the valid range
            if 0 <= type_event < dim_process:
                buckets[i][type_event].append(event['time_since_start'])
    
    # Convert event times to numpy arrays for each bucket
    return [[np.array(times, dtype=np.float64) for times in sublist] for sublist in buckets]


def process_dataset(config_path, dataset_name: str = 'earthquake') -> Dict[str, Dict]:
    """
    Full data processing pipeline.

    Returns:
        Dict[str, Dict]: A dictionary containing processed 'train', 'valid', and 'test' datasets.
    """
    # Load raw data
    raw_data = load_raw_data(config_path, dataset_name)
    
    # Extract event dimension from the training data
    dim_process = raw_data["train"]["dim_process"]
    
    # Process each data split (train, valid, test)
    return {
        "train": {
            "sequences": extract_sequences(raw_data["train"]["train"], dim_process),
            "buckets": create_event_buckets(raw_data["train"]["train"], dim_process)
        },
        "valid": {
            "sequences": extract_sequences(raw_data["dev"]["dev"], dim_process),
            "buckets": create_event_buckets(raw_data["dev"]["dev"], dim_process)
        },
        "test": {
            "sequences": extract_sequences(raw_data["test"]["test"], dim_process),
            "buckets": create_event_buckets(raw_data["test"]["test"], dim_process)
        }
    }