import pickle
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any
from rep_tpp.data_loader import load_raw_data


def merge_and_resplit_tpp_data(
    data: Dict[str, Dict[str, Any]],
    dev_split: float,
    test_split: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge train/dev/test sequences and then resplit each sequence by time.

    Input format (as you described):
        data = {
          "train": {"dim_process": int, "train": List[List[Dict]]},
          "dev": {"dim_process": int, "dev": List[List[Dict]]},
          "test":  {"dim_process": int, "test":  List[List[Dict]]},
        }

    Split rule for each sequence:
        final_time = last_event["time_since_start"]
        new_train_seq: events with time_since_start <= final_time * dev_split
        new_dev_seq: events with time_since_start <= final_time * test_split
        new_test_seq : full sequence

    Returns:
        Dict with the same outer structure as input.
    """
    # ---- validations ----
    if not (0.0 < dev_split <= 1.0):
        raise ValueError(f"dev_split must be in (0, 1], got {dev_split}")
    if not (0.0 < test_split <= 1.0):
        raise ValueError(f"test_split must be in (0, 1], got {test_split}")
    if dev_split > test_split:
        raise ValueError(f"Require dev_split <= test_split, got {dev_split} > {test_split}")

    for split in ("train", "dev", "test"):
        if split not in data:
            raise KeyError(f"Missing top-level key: {split}")
        if "dim_process" not in data[split]:
            raise KeyError(f"Missing dim_process in data['{split}']")
        if split not in data[split]:
            raise KeyError(f"Missing sequences key '{split}' in data['{split}']")

    dim_process = int(data["train"]["dim_process"])
    
    if int(data["dev"]["dim_process"]) != dim_process or int(data["test"]["dim_process"]) != dim_process:
        raise ValueError("Inconsistent dim_process across splits.")

    # ---- merge all sequences ----
    all_seqs: List[List[Dict[str, Any]]] = []
    all_seqs.extend(data["train"]["train"])
    all_seqs.extend(data["dev"]["dev"])
    all_seqs.extend(data["test"]["test"])

    # ---- resplit per sequence ----
    new_train: List[List[Dict[str, Any]]] = []
    new_dev: List[List[Dict[str, Any]]] = []
    new_test: List[List[Dict[str, Any]]] = []

    for seq in all_seqs:
        if not seq:
            continue

        seq_sorted = sorted(seq, key=lambda e: float(e["time_since_start"]))

        final_time = float(seq_sorted[-1]["time_since_start"])
        first_time = float(seq_sorted[0]["time_since_start"])
        train_cut = (final_time - first_time) * dev_split + first_time
        dev_cut = (final_time - first_time) * test_split + first_time

        train_seq = [e for e in seq_sorted if float(e["time_since_start"]) <= train_cut]
        dev_seq = [e for e in seq_sorted if float(e["time_since_start"]) <= dev_cut]
        test_seq = seq_sorted  # full

        new_train.append(train_seq)
        new_dev.append(dev_seq)
        new_test.append(test_seq)

    # ---- pack back to EasyTPP-like dict ----
    return {
        "train": {"dim_process": dim_process, "train": new_train},
        "dev": {"dim_process": dim_process, "dev": new_dev},
        "test":  {"dim_process": dim_process, "test":  new_test},
    }


def save_resplit_tpp_data(data_resplit: dict, base_dir: str):
    """
    Save resplit TPP data to train/dev/test pkl files.

    Files saved:
        {base_dir}/train.pkl
        {base_dir}/dev.pkl
        {base_dir}/test.pkl
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    save_map = {
        "train": "train.pkl",
        "dev": "dev.pkl",   # EasyTPP uses 'dev'
        "test":  "test.pkl",
    }

    for split, filename in save_map.items():
        out_path = base_dir / filename

        with out_path.open("wb") as f:
            pickle.dump(data_resplit[split], f)

        print(f"[Saved] {split} -> {out_path}")


def data_resplit(dataset_name, resplit_points):
    data_config_path = f"./config/{dataset_name}_data_config.yaml"

    data = load_raw_data(data_config_path, dataset_name)
    data_resplit = merge_and_resplit_tpp_data(data, resplit_points[0], resplit_points[1])
    save_resplit_tpp_data(data_resplit, f"./data/{dataset_name}/resplit")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_dir}", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    resplit_points = config["data_resplit_config"]["resplit_points"]

    data_resplit(dataset_name, resplit_points)