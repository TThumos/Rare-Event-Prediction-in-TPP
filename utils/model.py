import torch
from torch import nn
import torch.nn.functional as F


def build_sample_residual_feature(batch, feature, sample_time):
    time_seqs, _, _, seq_non_pad_mask, _ = batch.values()
    B, _ = time_seqs.shape
    _, sample_per_seq = sample_time.shape

    temp_time_seqs = time_seqs.clone().to(dtype=time_seqs.dtype)
    inf = float('inf')
    temp_time_seqs = temp_time_seqs.masked_fill(~seq_non_pad_mask, inf)

    sample_index = ((sample_time.unsqueeze(2) - temp_time_seqs.unsqueeze(1)) >= 0).sum(dim = 2) - 1
    row_index = torch.arange(B, device=sample_time.device).unsqueeze(1).repeat(1, sample_per_seq)
    sample_time_seqs = temp_time_seqs[row_index, sample_index]
    sample_delta_time = sample_time - sample_time_seqs
    sample_delta_time[sample_delta_time == -float("inf")] = 0.0
    sample_feature = feature[row_index, sample_index]

    return sample_feature, sample_delta_time


def build_sample_hawkes_feature(sample_hawkes_time, feature, sample_time):
    B, _ = sample_hawkes_time.shape
    _, sample_per_seq = sample_time.shape

    sample_index = ((sample_time.unsqueeze(2) - sample_hawkes_time.unsqueeze(1)) >= 0).sum(dim = 2) - 1
    row_index = torch.arange(B, device=sample_time.device).unsqueeze(1).repeat(1, sample_per_seq)
    sample_feature = feature[row_index, sample_index]

    return sample_feature


def build_sample_label(batch, sample_time, num_event_types, tau, device):
    time_seqs, _, type_seqs, seq_non_pad_mask, _ = batch.values()
    B, _ = time_seqs.shape
    _, sample_per_seq = sample_time.shape

    times_for_future = time_seqs.clone().to(dtype=time_seqs.dtype)
    inf = float('inf')
    times_for_future = times_for_future.masked_fill(~seq_non_pad_mask, inf)

    times_i = sample_time.unsqueeze(2)
    times_j = times_for_future.unsqueeze(1)
    future_mask = (times_j > times_i) & (times_j <= (times_i + tau))

    label = torch.zeros((B, sample_per_seq, num_event_types), dtype=torch.int, device=device)
    b_idx, i_idx, j_idx = torch.where(future_mask)  # 1D tensors (num_pairs,)

    if b_idx.numel() > 0:
        types_at_j = type_seqs[b_idx, j_idx]
        real_mask = (types_at_j != num_event_types)
        if real_mask.any():
            b_idx = b_idx[real_mask]
            i_idx = i_idx[real_mask]
            types_at_j = types_at_j[real_mask]
            label[b_idx, i_idx, types_at_j] = 1

    return label


def build_sample(batch, hidden_state, num_event_types, split_points, tau, sample_per_seq, device):
    origin_batch, kept_batch, residual_batch, extra = \
        batch['origin'], batch['kept'], batch['residual'], batch['extra']

    time_seqs, _, type_seqs, seq_non_pad_mask, _ = origin_batch.values()
    B, _ = time_seqs.shape

    times_for_final = time_seqs.clone().to(dtype=time_seqs.dtype)
    neg_inf = float('-inf')
    times_for_final = times_for_final.masked_fill(~seq_non_pad_mask, neg_inf)
    first_time = times_for_final[:, 0]
    final_time, _ = times_for_final.max(dim=1)

    # sample_time
    lower_bounds = first_time + (final_time - first_time)*split_points[0] - tau
    lower_bounds = torch.clamp(lower_bounds, min=first_time)
    upper_bounds = first_time + (final_time - first_time)*split_points[1] - tau
    upper_bounds = torch.clamp(upper_bounds, min=first_time)
    uniform_rand = torch.rand(B, sample_per_seq, device=device)
    sample_time = uniform_rand * (upper_bounds.unsqueeze(1) - lower_bounds.unsqueeze(1)) + lower_bounds.unsqueeze(1)

    # sample_delta_time & sample_hidden_state
    if (kept_batch is not None) & (extra is not None):
        hawkes_sample_time, hawkes_feature = extra[..., 0], extra[..., 1:]
        sample_feature_kept = build_sample_hawkes_feature(hawkes_sample_time, hawkes_feature, sample_time)
    else:
        sample_feature_kept = None
    sample_feature_residual, sample_delta_time_residual = build_sample_residual_feature(residual_batch, hidden_state, sample_time)

    # label
    label = build_sample_label(origin_batch, sample_time, num_event_types, tau, device)

    return sample_feature_kept, sample_feature_residual, sample_delta_time_residual, label


class MultiHeadMLP(nn.Module):
    def __init__(self, layer_dims, num_heads):
        super(MultiHeadMLP, self).__init__()
        self.num_heads = num_heads
        
        assert len(layer_dims) >= 2
        assert layer_dims[-1] == 2
        
        self.mlps = nn.ModuleList()
        for _ in range(num_heads):
            layers = []
            for i in range(len(layer_dims) - 1):
                in_dim = layer_dims[i]
                out_dim = layer_dims[i + 1]
                
                layers.append(nn.Linear(in_dim, out_dim, bias=False))
                
                if i < len(layer_dims) - 2:
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.Softmax(dim=-1))
            
            self.mlps.append(nn.Sequential(*layers))
    
    def forward(self, x):
        outputs = []
        
        for mlp in self.mlps:
            mlp_output = mlp(x)
            slice_output = mlp_output[..., 0]
            outputs.append(slice_output)
        
        final_output = torch.stack(outputs, dim=-1)
        
        return final_output


class RareEventTPPModel(torch.nn.Module):
    def __init__(self, base_model, base_model_id, rare_head, freq, weight, kwargs):
        super(RareEventTPPModel, self).__init__()
        self.base_model = base_model
        self.base_model_id = base_model_id
        self.rare_head = rare_head
        self.freq = freq
        self.weight = weight

        self.num_event_types = kwargs["num_event_types"]

        split_points = kwargs["split_points"]
        self.valid_split_point, self.test_split_point = split_points[0], split_points[1]

        self.tau = kwargs["tau"]

        sample_per_seq = kwargs["sample_per_seq"]
        self.sample_per_seq_train, self.sample_per_seq_valid, self.sample_per_seq_test = sample_per_seq[0], sample_per_seq[1], sample_per_seq[2]

        alpha_range = kwargs["alpha_range"]
        self.alpha_lower, self.alpha_upper = alpha_range[0], alpha_range[1]

        self.device = kwargs["device"]

        self.stage = None

    def forward(self, batch):
        if self.base_model_id in ["THP", "AttNHP"]:
            hidden_state = self.base_model.forward(
                batch['residual']['time_seqs'], 
                batch['residual']['type_seqs'], 
                batch['residual']['attention_mask'],
            )
        if self.base_model_id in ["SAHP"]:
            hidden_state = self.base_model.forward(
                batch['residual']['time_seqs'],
                batch['residual']['time_delta_seqs'],
                batch['residual']['type_seqs'], 
                batch['residual']['attention_mask'],
            )
        if self.base_model_id in ["NHP", "RMTPP"]:
            _, hidden_state = self.base_model.forward(batch['residual'].values())
        
        if self.stage == "train":
            start_split_point, end_split_point = 0.0, 1.0
            sample_per_seq = self.sample_per_seq_train
        elif self.stage == "valid":
            start_split_point, end_split_point = self.valid_split_point/self.test_split_point, 1.0
            sample_per_seq = self.sample_per_seq_valid
        elif self.stage == "test":
            start_split_point, end_split_point = self.test_split_point, 1.0
            sample_per_seq = self.sample_per_seq_test
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        sample_feature_kept, sample_feature_residual, sample_delta_time_residual, label = build_sample(
            batch, hidden_state, self.num_event_types, (start_split_point, end_split_point), self.tau, sample_per_seq, self.device
        )
        label = label.float()

        if sample_feature_kept is not None:
            input_ls = [sample_feature_kept, sample_feature_residual, sample_delta_time_residual.unsqueeze(-1)]
        else:
            input_ls = [sample_feature_residual, sample_delta_time_residual.unsqueeze(-1)]
        input_tensor = torch.cat(input_ls, dim=-1)
        probs = self.rare_head(input_tensor)  # [B, N, K]
        return probs, label

    def loss(self, batch):
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
            elif value is not None:
                for k, v in value.items():
                    if torch.is_tensor(v):
                        batch[key][k] = v.to(self.device)
        probs, label = self.forward(batch)  # [B, N, K], [B, N, K]
        B, N, K = label.shape

        alphas = torch.rand(B, device=self.device) * (self.alpha_upper - self.alpha_lower) + self.alpha_lower  # [B]
        rare_mask = (self.freq.unsqueeze(0) < alphas.unsqueeze(1))  # [B, K]
        rare_mask_expand = rare_mask.unsqueeze(1).expand(-1, N, -1)  # [B, N, K]

        loss_matrix = F.binary_cross_entropy(probs, label, reduction='none')  # [B, N, K]

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        bce_loss = (loss_matrix * rare_mask_expand.float() * weight).sum()
        num_rare = (rare_mask_expand * weight).sum()

        loglike_loss, num_events = self.base_model.loglike_loss(batch['residual'].values())
        
        return bce_loss, num_rare, loglike_loss, num_events