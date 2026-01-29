# red.py
import numpy as np
import pickle
import yaml
import argparse
from pathlib import Path

from rep_tpp.data_process import process_dataset
from rep_tpp.residual_filter import ResidualDataProcessor


def red(dataset_name, num_event_types, dataset_weights, threshold):
    # Load and process data (moved inside)
    resplit_config_path = f"./config/{dataset_name}_resplit_data_config.yaml"
    processed_data = process_dataset(resplit_config_path, dataset_name)

    kept_ratio = np.mean(np.concatenate(dataset_weights['train']) > threshold)
    print(f"Kept ratio: {kept_ratio*100: .2f}%")

    output_dir_residual = f"./data/{dataset_name}/residual"
    output_dir_kept = f"./data/{dataset_name}/kept"

    processor = ResidualDataProcessor(threshold=threshold)

    processor.process_and_save(
        weights=dataset_weights,
        raw_datasets={
            'train': processed_data['train']['sequences'],
            'valid': processed_data['valid']['sequences'],
            'test': processed_data['test']['sequences']
        },
        output_dir_residual=Path(output_dir_residual),
        output_dir_kept=Path(output_dir_kept),
        dim_process=num_event_types
    )

    return kept_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_dir}", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    num_event_types = config["num_event_types"]
    threshold = config["red_config"]["threshold"]

    with open(f"./weight/{dataset_name}.pkl", "rb") as f:
        dataset_weights = pickle.load(f)

    red(dataset_name, num_event_types, dataset_weights, threshold)