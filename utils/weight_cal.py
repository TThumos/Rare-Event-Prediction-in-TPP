import numpy as np
import pickle
import yaml
import argparse
from tqdm import tqdm

from rep_tpp.data_loader import load_raw_data
from rep_tpp.data_process import process_dataset
from rep_tpp.hawkes import HawkesModelHandler
from rep_tpp.weight_compute import WeightCalculator, WeightAnalyzer


def weight_cal(dataset_name, processed_data, num_event_types, weight_config, max_iter):

    # Initialize calculator
    calculator = WeightCalculator(weight_config)

    # Model initialization and training
    handler = HawkesModelHandler(max_iter=max_iter, random_seed=42)
    decay_matrix = handler.create_decay_matrix(num_event_types, 4, 0.01)

    model = handler.initialize_model()
    trained_model = handler.train_model(
        model, 
        processed_data['train']['buckets']
    )

    # Weight computation
    dataset_weights = {
        'train': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['train']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['train']['sequences']['time_seqs']),
                desc="Processing train set",
                total=len(processed_data['train']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ],
        'valid': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['valid']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['valid']['sequences']['time_seqs']),
                desc="Processing valid set",
                total=len(processed_data['valid']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ],
        'test': [
            calculator.compute_w(
                baseline=trained_model.baseline,
                t_points=seq,
                adjacency_matrix=trained_model.adjacency,
                decay_matrix=trained_model.decays,
                event_times=processed_data['test']['buckets'][i]
            )
            for i, seq in tqdm(enumerate(processed_data['test']['sequences']['time_seqs']),
                desc="Processing test set",
                total=len(processed_data['test']['sequences']['time_seqs']),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        ]
    }

    all_weights = []
    all_weights.extend(dataset_weights['train'])
    all_weights.extend(dataset_weights['valid'])
    all_weights.extend(dataset_weights['test'])

    # Result analysis
    total_ratio, ratios = WeightAnalyzer.compute_zero_ratio(all_weights)
    print(WeightAnalyzer.format_result(total_ratio, [
        np.mean(np.concatenate(dataset_weights['train']) == 0),
        np.mean(np.concatenate(dataset_weights['valid']) == 0),
        np.mean(np.concatenate(dataset_weights['test']) == 0)
    ]))

    with open(f"weight/{dataset_name}.pkl", "wb") as f:
        pickle.dump(dataset_weights, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_dir}", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    weight_config = config["weight_cal_config"]["weight_config"]
    max_iter = config["weight_cal_config"]["max_iter"]

    # Load and process data
    resplit_config_path = f"./config/{dataset_name}_resplit_data_config.yaml"
    raw_data = load_raw_data(resplit_config_path, dataset_name)
    processed_data = process_dataset(resplit_config_path, dataset_name)
    num_event_types = raw_data["train"]["dim_process"]

    dataset_weights = weight_cal(dataset_name, processed_data, num_event_types, weight_config, max_iter)