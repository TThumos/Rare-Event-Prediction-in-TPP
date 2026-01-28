import os
import numpy as np
import torch
import pickle
import yaml
import argparse

from residual_tpp.data_process import process_dataset
from residual_tpp.hawkes import HawkesModelHandler


def intensity_cal(
    dataset_name: str,
    num_event_types: int,
    max_iter: int,
    sample_per_seq: int,
    kept_ratio: float,
    threshold: float,
    reuse: bool = True,
    out_dir: str = "intensity",
):
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{dataset_name}_kr{int(kept_ratio):03d}_th{threshold:.3f}.pkl"
    )

    if reuse and os.path.exists(out_path):
        print(f"[intensity_cal] Reuse: {out_path}")
        return out_path

    # Train Hawkes
    handler = HawkesModelHandler(max_iter=max_iter, random_seed=42)
    _ = handler.create_decay_matrix(num_event_types, 4, 0.01)
    model = handler.initialize_model()

    processed_data = process_dataset(f"./config/{dataset_name}_resplit_data_config.yaml", dataset_name)
    train = processed_data["train"]["buckets"]
    train = [sub for sub in train if not all((len(a) == 0) for a in sub)]
    trained_model = handler.train_model(model, train)
    handler.model = trained_model

    processed_kept_data = process_dataset(f"./config/{dataset_name}_kept_data_config.yaml", dataset_name)
    test_kept = processed_kept_data["test"]["buckets"]

    train_data = processed_data["train"]["sequences"]
    valid_data = processed_data["valid"]["sequences"]
    test_data = processed_data["test"]["sequences"]

    max_length = max(
        max(len(seq) for seq in train_data["time_delta_seqs"]),
        max(len(seq) for seq in valid_data["time_delta_seqs"]),
        max(len(seq) for seq in test_data["time_delta_seqs"]),
    )

    time_seq = np.full((len(test_data["time_seqs"]), max_length), num_event_types, dtype=float)
    for i, time_seq_i in enumerate(test_data["time_seqs"]):
        time_seq[i, : len(time_seq_i)] = time_seq_i

    time_seq = torch.tensor(time_seq, dtype=torch.float32)

    times_for_final = time_seq.clone()
    neg_inf = float("-inf")
    times_for_final = times_for_final.masked_fill(time_seq == num_event_types, neg_inf)
    first_time = times_for_final[:, 0]
    final_time, _ = times_for_final.max(dim=1)

    uniform_rand = torch.rand(time_seq.shape[0], sample_per_seq)
    uniform_rand, _ = torch.sort(uniform_rand, dim=1)
    sample_time = uniform_rand * (final_time.unsqueeze(1) - first_time.unsqueeze(1)) + first_time.unsqueeze(1)

    intensities = handler.compute_event_intensities(sample_time, test_kept)
    output = torch.concat([sample_time.unsqueeze(-1), intensities], dim=-1)

    payload = {
        "meta": {
            "dataset_name": dataset_name,
            "num_event_types": int(num_event_types),
            "threshold": float(threshold),
            "sample_per_seq": int(sample_per_seq),
        },
        "data": output,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[intensity_cal] Saved: {out_path}")
    return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_dir}", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    num_event_types = config["num_event_types"]
    max_iter = config["intensity_cal_config"]["max_iter"]
    sample_per_seq =  config["intensity_cal_config"]["sample_per_seq"]
    kept_ratio = config["intensity_cal_config"]["kept_ratio"]
    threshold = config["red_config"]["threshold"]
    
    intensity_cal(
        dataset_name,
        num_event_types,
        max_iter,
        sample_per_seq,
        kept_ratio,
        threshold,
        reuse = True,
        out_dir = "intensity",
    )