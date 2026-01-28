import os
import random
import argparse
import yaml
import pickle
import numpy as np

from utils.red import red
from utils.intensity_cal import intensity_cal
from utils.runner import runner, DictToObject

def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception:
        pass


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config['dataset_name']
    num_event_types = config['num_event_types']
    alpha_range = config['runner_config']['alpha_range']

    weight_path = f"./weight/{dataset_name}.pkl"
    with open(weight_path, 'rb') as f:
        dataset_weights = pickle.load(f)

    train_weights = np.concatenate(dataset_weights['train'])
    zero_weight_ratio = (train_weights == 0).sum()/len(train_weights)
    target_kept_ratios = [5]
    i = 10
    while i < (1 - zero_weight_ratio)*100:
        target_kept_ratios.append(i)
        i += 5
    threshold_list = []
    for target_kept_ratio in target_kept_ratios:
        if target_kept_ratio == 0:
            threshold = 1 - 1e-6
        elif target_kept_ratio == 100:
            threshold = 0 + 1e-6
        else:
            q = 100 - target_kept_ratio
            threshold = float(np.percentile(train_weights, q))
        threshold_list.append(threshold)

    # intensity_cal_config
    max_iter = config['intensity_cal_config']['max_iter']
    sample_per_seq = config['intensity_cal_config']['sample_per_seq']
    # runner_config
    base_model_id = config['runner_config']['base_model_id']
    base_model_config_dict = config[base_model_id]
    base_model_config_dict['num_event_types'] = num_event_types
    base_model_config_dict['num_event_types_pad'] = num_event_types + 1
    base_model_config_dict['pad_token_id'] = num_event_types
    base_model_config_dict['thinning'] = DictToObject(base_model_config_dict['thinning'])
    base_model_config = DictToObject(base_model_config_dict)
    rare_head_config = config['runner_config']['rare_head_config']
    model_config = config['runner_config']['model_config']
    model_config['num_event_types'] = num_event_types
    optimizer_config = config['runner_config']['optimizer_config']
    batch_size = config['runner_config']['batch_size']

    f1_summary_dict = {kept_ratio: None for kept_ratio in [0] + target_kept_ratios}

    # origin
    rare_head_config_origin = rare_head_config.copy()
    rare_head_config_origin[0] = rare_head_config_origin[0] - num_event_types

    sub_result_path = f"./results/{dataset_name}_{base_model_id}_origin.pkl"
    if os.path.exists(sub_result_path):
        with open(sub_result_path, "rb") as f:
            (f1_summary_valid, f1_summary_test) = pickle.load(f)
    
    else:
        f1_summary_valid, f1_summary_test = runner(
            dataset_name,
            num_event_types,
            False,
            base_model_id,
            base_model_config,
            rare_head_config_origin,
            model_config,
            alpha_range,
            optimizer_config,
            None,
            batch_size,
            model_config['device'],
        )
        with open(sub_result_path, "wb") as f:
                pickle.dump((f1_summary_valid, f1_summary_test), f)
    
    f1_summary_dict[0] = (f1_summary_valid, f1_summary_test)

    # residual
    for i, (target_kept_ratio, threshold) in enumerate(zip(target_kept_ratios, threshold_list)):

        print(f"\n====== {dataset_name} {base_model_id} Sweep {i+1}/{len(target_kept_ratios)}: threshold = {threshold:.6f} ======")

        kept_ratio = red(dataset_name, num_event_types, dataset_weights, threshold)
        kept_ratio_percent = kept_ratio * 100

        sub_result_path = f"./results/{dataset_name}_{base_model_id}_kr{int(kept_ratio_percent):03d}.pkl"

        if os.path.exists(sub_result_path):
            with open(sub_result_path, "rb") as f:
                (f1_summary_valid, f1_summary_test) = pickle.load(f)

        else:
            intensity_path = intensity_cal(
                dataset_name,
                num_event_types,
                max_iter,
                sample_per_seq,
                kept_ratio_percent,
                threshold,
                reuse=True,
            )

            f1_summary_valid, f1_summary_test = runner(
                dataset_name,
                num_event_types,
                True,
                base_model_id,
                base_model_config,
                rare_head_config,
                model_config,
                alpha_range,
                optimizer_config,
                intensity_path,
                batch_size,
                model_config['device'],
            )
            with open(sub_result_path, "wb") as f:
                pickle.dump((f1_summary_valid, f1_summary_test), f)

        f1_summary_dict[target_kept_ratio] = (f1_summary_valid, f1_summary_test)

    result_path = f"./results/{dataset_name}_{base_model_id}.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(f1_summary_dict, f)


config_dict = {
    'earthquake': {
        'num_event_types': 7,
        'rare_head_config': [40, 32, 16, 2],
        'tau': 2,
        'hidden_size': [32, 8, 32, 32, 8],
    },
    'taxi': {
        'num_event_types': 10,
        'rare_head_config': [43, 32, 16, 2],
        'tau': 1,
        'hidden_size': [32, 8, 32, 32, 8],
    },
    'retweet': {
        'num_event_types': 3,
        'rare_head_config': [36, 32, 16, 2],
        'tau': 50,
        'hidden_size': [32, 8, 32, 32, 8],
    },
    'stackoverflow': {
        'num_event_types': 22,
        'rare_head_config': [87, 64, 16, 2],
        'tau': 6,
        'hidden_size': [64, 16, 64, 64, 16],
    },
    'taobao': {
        'num_event_types': 17,
        'rare_head_config': [82, 64, 16, 2],
        'tau': 4,
        'hidden_size': [64, 16, 64, 64, 16],
    },
    'amazon': {
        'num_event_types': 16,
        'rare_head_config': [81, 64, 16, 2],
        'tau': 2,
        'hidden_size': [64, 16, 64, 64, 16],
    },
    'iptv': {
        'num_event_types': 16,
        'rare_head_config': [81, 32, 16, 2],
        'tau': 3,
        'hidden_size': [64, 16, 64, 64, 16],
    },
}


def build_yaml_content(dataset_name, base_model_id):

    config = config_dict[dataset_name]

    num_event_types = config['num_event_types']
    rare_head_config = config['rare_head_config']
    tau = config['tau']
    hidden_size = config['hidden_size']

    yaml_content = f"""
dataset_name: {dataset_name}
num_event_types: {num_event_types}

data_resplit_config:
  resplit_points: [0.6, 0.8]

weight_cal_config:
  weight_config: 
    a: 4.0
    b: 16.0
    rho1: 8.0
    rho2: 8.0
  max_iter: 500

red_config:
  # threshold: 

intensity_cal_config:
  max_iter: 500
  sample_per_seq: 1000
  # kept_ratio: 

runner_config:
  # if_residual:
  base_model_id: {base_model_id}
  rare_head_config: {rare_head_config}
  alpha_range: [0, 0.1]
  model_config:
    split_points: [0.6, 0.8]
    tau: {tau}
    sample_per_seq: [1024, 512, 512]
    alpha_range: [0.1, 0.1]
    device: cuda
  optimizer_config:
    lr: 0.01
    factor: 0.5
    patience: 2
    threshold: 0.0001
    min_lr: 0.0001
    max_epoch: 300
  batch_size: 256 # iptv 32

RMTPP:
  hidden_size:  {hidden_size[0]}
  loss_integral_num_sample_per_step: 20
  thinning:
    num_seq: 10
    num_sample: 1
    num_exp: 500 
    look_ahead_time: 10
    patience_counter: 5 
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 5
    num_step_gen: 1
  gpu: 0
  use_mc_samples: True

NHP:
  model_specs: {{}}
  hidden_size: {hidden_size[1]}
  loss_integral_num_sample_per_step: 20
  thinning:
    num_seq: 10
    num_sample: 1
    num_exp: 500 
    look_ahead_time: 10
    patience_counter: 5 
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 5
    num_step_gen: 1
  gpu: 0
  use_mc_samples: True

SAHP:
  hidden_size: {hidden_size[2]}
  time_emb_size: 16
  use_ln: False
  num_layers: 4
  num_heads: 4
  dropout_rate: 0.0
  loss_integral_num_sample_per_step: 20
  thinning: 
    num_seq: 10
    num_sample: 1
    num_exp: 500 
    look_ahead_time: 10
    patience_counter: 5 
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 5
    num_step_gen: 1
  gpu: 0
  use_mc_samples: True

THP:
  hidden_size: {hidden_size[3]}
  time_emb_size: 16
  use_ln: False
  num_layers: 4
  num_heads: 4
  dropout_rate: 0.0
  loss_integral_num_sample_per_step: 20
  thinning: 
    num_seq: 10
    num_sample: 1
    num_exp: 500 
    look_ahead_time: 10
    patience_counter: 5 
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 5
    num_step_gen: 1
  gpu: 0
  use_mc_samples: True

AttNHP:
  hidden_size: {hidden_size[4]}
  time_emb_size: 4
  num_layers: 4
  num_heads: 4
  use_ln: False
  dropout_rate: 0.0
  loss_integral_num_sample_per_step: 20
  thinning: 
    num_seq: 10
    num_sample: 1
    num_exp: 500 
    look_ahead_time: 10
    patience_counter: 5 
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 5
    num_step_gen: 1
  gpu: 0
  use_mc_samples: True
    """

    with open("temp_config.yaml", "w") as file:
        file.write(yaml_content)


if __name__ == "__main__":
    seed_everything(100)
    for dataset_name in ['earthquake', 'stackoverflow', 'amazon', 'taobao', 'iptv']: # ['earthquake', 'taxi', 'retweet', 'stackoverflow', 'taobao', 'amazon', 'iptv']
        for base_model_id in ['RMTPP', 'NHP', 'SAHP', 'THP', 'AttNHP']: # ['RMTPP', 'NHP', 'SAHP', 'THP', 'AttNHP']
            print(f"========== {dataset_name} {base_model_id} ==========")
            build_yaml_content(dataset_name, base_model_id)
            main()
