from typing import Tuple, Optional, Dict, Union, List
import numpy as np
import torch
from torch import Tensor
from tick.hawkes import HawkesExpKern


class HawkesModelHandler:
    def __init__(self, max_iter: int = 500, random_seed: Optional[int] = 42):
        """Initialize the Hawkes process model handler
        
        Args:
            max_iter: The maximum number of iterations, default is 500
            random_seed: The random seed for reproducibility, default is 42 (None disables it)
        """
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.model = None
        self.decay_matrix = None
        self._validate_init_params()

    
    def _validate_init_params(self):
        """Validate initialization parameters"""
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be integer or None, got {type(self.random_seed)}")


    def create_decay_matrix(
        self,
        num_event_types: int,
        base_value: float = 1.0,
        noise_scale: float = 0.01
    ) -> np.ndarray:
        """Create a decay matrix (identity matrix + Gaussian noise)
        
        Args:
            num_event_types: The number of event types
            base_value: The base value for the diagonal, default is 1.0
            noise_scale: The noise scaling factor, default is 0.01
            
        Returns:
            np.ndarray: Decay matrix of shape (num_event_types, num_event_types)
        """
        if num_event_types <= 0:
            raise ValueError(f"Invalid number of event types num_event_types={num_event_types}, must be a positive integer.")
        if base_value <= 0:
            raise ValueError(f"Base value must be positive, got base_value={base_value}")

        identity = np.eye(num_event_types) * base_value
        noise = noise_scale * np.random.randn(num_event_types, num_event_types)
        self.decay_matrix = identity + noise
        return self.decay_matrix

    
    def initialize_model(self) -> HawkesExpKern:
        """Initialize the Hawkes process model
        
        Returns:
            HawkesExpKern: The initialized Hawkes model
        """
        if self.decay_matrix is None:
            raise ValueError("Decay matrix has not been created. Please call create_decay_matrix() first.")

        return HawkesExpKern(decays=self.decay_matrix, max_iter=self.max_iter)

    
    def train_model(self, model: HawkesExpKern, train_data: list) -> HawkesExpKern:
        """Train the Hawkes process model
        
        Args:
            model: The initialized Hawkes model
            train_data: The training data in the format expected by the tick library
            
        Returns:
            HawkesExpKern: The trained Hawkes model
        """
        model.fit(train_data)
        return model
    

    def _validate_model_initialized(self):
        """Ensure model parameters are available"""
        if not all([hasattr(self.model, attr) for attr in ['baseline', 'adjacency', 'decays']]):
            raise RuntimeError("Model parameters not initialized. Train model first.")
    

    def compute_event_intensities(self, time_seqs: Tensor, event_times: List[List[List[float]]]) -> Tensor:
        """
        time_seqs: [B, Q]，要求每个 b 内部升序（你已在采样时 sort）
        event_times[b][j]: 类型 j 的历史事件时间列表（建议升序；不升序我这里会 sort）
        return: [B, Q, K]
        """
        self._validate_model_initialized()

        baseline = np.asarray(self.model.baseline, dtype=np.float64)      # [K]
        adjacency = np.asarray(self.model.adjacency, dtype=np.float64)    # [K,K]
        decays = np.asarray(self.model.decays, dtype=np.float64)          # [K,K]

        B, Q = time_seqs.shape
        K = baseline.shape[0]

        tq_all = time_seqs.detach().cpu().numpy().astype(np.float64)      # [B,Q]
        out = np.zeros((B, Q, K), dtype=np.float64)

        for b in range(B):
            tq = tq_all[b]  # [Q] 升序

            # 合并该序列所有事件 (time, type) 并排序
            ev_t = []
            ev_type = []
            for j in range(K):
                arr = event_times[b][j]
                if arr is None or arr.size == 0:
                    continue
                for tt in arr:
                    ev_t.append(float(tt))
                    ev_type.append(j)

            if len(ev_t) == 0:
                out[b] = baseline[None, :].repeat(Q, axis=0)
                continue

            ev_t = np.asarray(ev_t, dtype=np.float64)
            ev_type = np.asarray(ev_type, dtype=np.int64)
            idx = np.argsort(ev_t)
            ev_t = ev_t[idx]
            ev_type = ev_type[idx]
            N = ev_t.shape[0]

            # 递推状态 S[k,j]
            S = np.zeros((K, K), dtype=np.float64)
            e_ptr = 0
            last_time = tq[0]

            # 先加入 last_time 之前的事件
            while e_ptr < N and ev_t[e_ptr] <= last_time:
                j = ev_type[e_ptr]
                S[:, j] += 1.0
                e_ptr += 1

            for qi in range(Q):
                t_now = tq[qi]

                # 推进到所有 <= t_now 的事件
                while e_ptr < N and ev_t[e_ptr] <= t_now:
                    t_e = ev_t[e_ptr]
                    dt = t_e - last_time
                    if dt > 0:
                        S *= np.exp(-decays * dt)
                        last_time = t_e
                    j = ev_type[e_ptr]
                    S[:, j] += 1.0
                    e_ptr += 1

                # 推进到 query time
                dtq = t_now - last_time
                if dtq > 0:
                    S *= np.exp(-decays * dtq)
                    last_time = t_now

                out[b, qi] = baseline + (adjacency * S).sum(axis=1)

        return torch.from_numpy(out).to(dtype=torch.float32, device=time_seqs.device)
    

    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,       # [B, L]
        time_delta_seqs: torch.Tensor, # [B, L] (unused in Hawkes intensities)
        type_seqs: torch.Tensor,       # [B, L]
        sample_dtimes: torch.Tensor,   # [B, L, S]
        compute_last_step_only: bool = False,
    ) -> torch.Tensor:
        """
        Compute Hawkes intensities λ(t) at sampled offsets for each sequence.

        Returns:
            [B, L, S, K] or [B, 1, S, K] if compute_last_step_only=True
        """
        self._validate_model_initialized()

        B, L = time_seqs.shape
        S = sample_dtimes.size(2)
        K = int(np.asarray(self.model.baseline).shape[0])

        if compute_last_step_only:
            out = torch.zeros((B, 1, S, K), dtype=torch.float32)
            l_list = [L - 1]
        else:
            out = torch.zeros((B, L, S, K), dtype=torch.float32)
            l_list = list(range(L))
        
        event_times_batch = []
        for b in range(B):
            buckets = [[] for _ in range(K)]
            for l in range(L):
                tp = int(type_seqs[b, l].item())
                t = float(time_seqs[b, l].item())
                buckets[tp].append(t)
            event_times_batch.append([np.asarray(x, dtype=np.float64) for x in buckets])

        for b in range(B):
            for l in l_list:
                t_base = time_seqs[b, l]                 # scalar tensor
                offsets = sample_dtimes[b, l, :]         # [S]
                query_times = (t_base + offsets)[None, :]  # [1, S]

                lambdas_1sk = self.compute_event_intensities(
                    time_seqs=query_times,
                    event_times=[event_times_batch[b]]
                )  # [1, S, K] (Tensor)

                if compute_last_step_only:
                    out[b, 0, :, :] = lambdas_1sk[0]
                else:
                    out[b, l, :, :] = lambdas_1sk[0]

        return out