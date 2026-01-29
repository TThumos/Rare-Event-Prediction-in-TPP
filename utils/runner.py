import numpy as np
import torch
import pickle
import yaml
import argparse
from easy_tpp.config_factory import Config
from easy_tpp.preprocess.data_loader import TPPDataLoader
from easy_tpp.model.torch_model.torch_thp import THP
from easy_tpp.model.torch_model.torch_nhp import NHP
from easy_tpp.model.torch_model.torch_attnhp import AttNHP
from easy_tpp.model.torch_model.torch_rmtpp import RMTPP
from easy_tpp.model.torch_model.torch_sahp import SAHP
from utils.data_loader import build_triple_loader_with_optional_extra
from utils.freq_cal import compute_freq
from utils.model import MultiHeadMLP, RareEventTPPModel
from utils.learner import train_one_epoch, evaluate

class DictToObject:
    def __init__(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)


def runner(dataset_name, num_event_types, if_residual, base_model_id, base_model_config, rare_head_config, model_config, alpha_range, optimizer_config, intensity_path, batch_size, device, if_weight=True):
    # Data loader
    # loader origin
    data_config_origin = Config.build_from_yaml_file(f'./config/{dataset_name}_resplit_data_config.yaml')
    tpp_loader_origin = TPPDataLoader(data_config_origin)
    tpp_loader_origin.kwargs['batch_size'] = batch_size

    train_loader_origin = tpp_loader_origin.train_loader()
    valid_loader_origin = tpp_loader_origin.valid_loader()
    test_loader_origin = tpp_loader_origin.test_loader()

    if if_residual:
        # loader kept
        data_config_kept = Config.build_from_yaml_file(f'./config/{dataset_name}_kept_data_config.yaml')
        tpp_loader_kept = TPPDataLoader(data_config_kept)
        tpp_loader_kept.kwargs['batch_size'] = batch_size

        train_loader_kept = tpp_loader_kept.train_loader()
        valid_loader_kept = tpp_loader_kept.valid_loader()
        test_loader_kept = tpp_loader_kept.test_loader()

        # loader residual
        data_config_residual = Config.build_from_yaml_file(f'./config/{dataset_name}_residual_data_config.yaml')
        tpp_loader_residual = TPPDataLoader(data_config_residual)
        tpp_loader_residual.kwargs['batch_size'] = batch_size

        train_loader_residual = tpp_loader_residual.train_loader()
        valid_loader_residual = tpp_loader_residual.valid_loader()
        test_loader_residual = tpp_loader_residual.test_loader()

        # intensity
        with open(intensity_path, "rb") as f:
            obj = pickle.load(f)
            intensities = obj["data"]
    
        # merge
        train_loader = build_triple_loader_with_optional_extra(
            train_loader_origin, train_loader_kept, train_loader_residual, intensities
        )
        valid_loader = build_triple_loader_with_optional_extra(
            valid_loader_origin, valid_loader_kept, valid_loader_residual, intensities
        )
        test_loader = build_triple_loader_with_optional_extra(
            test_loader_origin, test_loader_kept, test_loader_residual, intensities
        )
    else:
        train_loader = build_triple_loader_with_optional_extra(
            train_loader_origin, None, train_loader_origin, None
        )
        valid_loader = build_triple_loader_with_optional_extra(
            valid_loader_origin, None, valid_loader_origin, None
        )
        test_loader = build_triple_loader_with_optional_extra(
            test_loader_origin, None, test_loader_origin, None
        )

    # Freq cal
    counts, total_events = compute_freq(valid_loader_origin, num_event_types)
    freq = (counts / total_events).to(device)

    # Model
    base_model_dict = {
        "NHP": NHP,
        "THP": THP,
        "AttNHP": AttNHP,
        "RMTPP": RMTPP,
        "SAHP": SAHP,
    }
    BaseModel = base_model_dict[base_model_id]
    base_model = BaseModel(base_model_config)
    base_model = base_model.to(device)

    rare_head = MultiHeadMLP(rare_head_config, num_event_types)
    rare_head = rare_head.to(device)

    alpha_lower = torch.tensor(alpha_range[0], device=device, dtype=freq.dtype)
    alpha_upper = torch.tensor(alpha_range[1], device=device, dtype=freq.dtype)
    weight = (alpha_upper - torch.maximum(alpha_lower, freq)) / (alpha_upper - alpha_lower)
    # ==================== 
    if if_weight == False:
        weight = torch.ones_like(weight)
    # ==================== 

    model = RareEventTPPModel(base_model, base_model_id, rare_head, freq, weight, model_config)

    # Train & Valid Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=optimizer_config["factor"], 
        patience=optimizer_config["patience"],
        threshold=optimizer_config["threshold"],
        min_lr=optimizer_config["min_lr"],
    )

    # best_loss = np.inf
    # bad_epochs = 0
    # max_bad_epochs = optimizer_config["max_bad_epochs"]
    epoch = 1

    while epoch <= optimizer_config["max_epoch"]:
        model.stage = "train"
        _ = train_one_epoch(model, train_loader, optimizer, (1/2, 1/2))

        train_loss, _ = evaluate(model, train_loader, alpha_range, weight=(1/2, 1/2), if_plot=False, compute_f1=False)

        scheduler.step(train_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}, LR: {lr:.2e}")

        # if valid_loss < best_loss - optimizer_config["threshold"]:
        #     best_loss = valid_loss
        #     bad_epochs = 0
        # else:
        #     bad_epochs += 1
        #     if bad_epochs >= max_bad_epochs:
        #         break

        epoch += 1
    
    model.stage = "valid"
    _, f1_summary_valid = evaluate(
        model, valid_loader, alpha_range,
        weight=(1/2, 1/2),
        if_plot=False,
        compute_f1=True,
        n_repeat=50,
        noise_std=1e-2
    )
    model.stage = "test"
    _, f1_summary_test = evaluate(
        model, test_loader, alpha_range,
        weight=(1/2, 1/2),
        if_plot=False,
        compute_f1=True,
        n_repeat=50,
        noise_std=1e-2
    )
    return f1_summary_valid, f1_summary_test



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(f"{args.config_dir}", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    num_event_types = config["num_event_types"]
    if_residual = config["runner_config"]["if_residual"]
    base_model_id = config["runner_config"]["base_model_id"]

    base_model_config = config[base_model_id]
    base_model_config["num_event_types"] = num_event_types
    base_model_config["num_event_types_pad"] = num_event_types + 1
    base_model_config["pad_token_id"] = num_event_types
    base_model_config["thinning"] = DictToObject(base_model_config["thinning"])
    base_model_config = DictToObject(base_model_config)

    rare_head_config = config["runner_config"]["rare_head_config"]

    model_config = config["runner_config"]["model_config"]
    model_config["num_event_types"] = num_event_types

    alpha_range = config["runner_config"]["alpha_range"]

    optimizer_config = config["runner_config"]["optimizer_config"]

    kept_ratio = config["intensity_cal_config"]["kept_ratio"]
    threshold = config["red_config"]["threshold"]
    intensity_path = f"intensity/{dataset_name}_kr{int(kept_ratio):03d}_th{threshold:.3f}.pkl"
    
    runner(dataset_name, num_event_types, if_residual, base_model_id, base_model_config, rare_head_config, model_config, alpha_range, optimizer_config, intensity_path)