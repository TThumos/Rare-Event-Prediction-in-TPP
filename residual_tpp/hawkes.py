from typing import Tuple, Optional, Dict, Union, List
import numpy as np
import os
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

    
    def evaluate_model(self, model: HawkesExpKern, datasets: dict) -> Tuple[float, float, float]:
        """Evaluate the model on different datasets
        
        Args:
            model: The trained Hawkes model
            datasets: A dictionary containing 'train', 'valid', and 'test' datasets
            
        Returns:
            Tuple: A tuple containing the scores for train, valid, and test datasets
        """
        scores = []
        for split in ['train', 'valid', 'test']:
            data = datasets.get(split)
            if not data:
                raise ValueError(f"Missing {split} dataset")
                
            score = model.score(
                events=data,
                baseline=model.baseline,
                adjacency=model.adjacency
            )
            scores.append(score)
            
        return tuple(scores)
    
    
    def compute_intensities(
        self,
        time_seqs: Tensor,
        time_delta_seqs: Tensor,
        type_seqs: Tensor,
        seq_mask: Tensor,
        event_times: List[List[List[float]]],
        n_samples: int = 30
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        
        """Compute Hawkes process intensities with multiple sampling strategies"""
        self._validate_model_initialized()

        # event event intensities
        event_intensities = self.compute_event_intensities(time_seqs[:,:-1], event_times)
        
        # Time delta sampling
        sampled_dtimes = self.sample_time_intervals(time_delta_seqs[:, 1:], n_samples)
        boundary_samples = self.sample_time_intervals(time_delta_seqs[:, :-1], 5)
        
        # Intensity calculations
        sampled_intensities = self.compute_sampled_intensities(sampled_dtimes, event_times)
        boundary_intensities = self.compute_sampled_intensities(boundary_samples, event_times)
        
        return {
            'event_intensities': event_intensities,
            'sampled_intensities': sampled_intensities,
            'processed_tensors': {
                'time_seq': time_seqs,
                'time_delta_seq': time_delta_seqs,
                'type_seq': type_seqs,
                'seq_mask': seq_mask,
                'sample_time_delta_seq': sampled_dtimes,
                'dtime_for_bound_sampled': boundary_samples,
                'bound_sampled_intensities': boundary_intensities
            }
        }   
    
    
    def _validate_model_initialized(self):
        """Ensure model parameters are available"""
        if not all([hasattr(self.model, attr) for attr in ['baseline', 'adjacency', 'decays']]):
            raise RuntimeError("Model parameters not initialized. Train model first.")

    
    # def compute_event_intensities(self, time_seqs: Tensor, event_times: List[List[List[float]]]) -> Tensor:
    #     """Exact intensity computation at event times"""
    #     batch_size, seq_len = time_seqs.shape
    #     num_types = len(self.model.baseline)
    #     intensities = torch.zeros((batch_size, seq_len, num_types))
        
    #     baseline = torch.tensor(self.model.baseline)
    #     adjacency = torch.tensor(self.model.adjacency)
    #     decays = torch.tensor(self.model.decays)

    #     for b in range(batch_size):
    #         for t in range(seq_len):
    #             current_time = time_seqs[b, t].item()
    #             for k in range(num_types):               
    #                 intensity = baseline[k].item()
    #                 for j in range(num_types):
    #                     for t_j in event_times[b][j]:
    #                         if t_j <= current_time:
    #                             delta = current_time - t_j
    #                             intensity += adjacency[k, j].item() * np.exp(-decays[k, j].item() * delta)
    #                 intensities[b, t, k] = intensity
    #     return intensities


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


    def sample_time_intervals(self, time_delta_seqs: Tensor, n_samples: int) -> Tensor:
        """Generate uniform samples within time intervals"""
        ratios = torch.linspace(0, 1, n_samples)
        return time_delta_seqs[:, :, None] * ratios  # [B, T-1, S]

    
    def compute_sampled_intensities(self, sampled_dtimes: Tensor, event_times: List[List[List[float]]]) -> Tensor:
        """Intensity computation at sampled time points"""
        batch_size, seq_len, n_samples = sampled_dtimes.shape
        num_types = len(self.model.baseline)
        intensities = torch.zeros(batch_size, seq_len, n_samples, num_types)
        
        baseline = torch.tensor(self.model.baseline)
        adjacency = torch.tensor(self.model.adjacency)
        decays = torch.tensor(self.model.decays)

        for b in range(batch_size):
            for t in range(seq_len):
                for s in range(n_samples):
                    current_time = sampled_dtimes[b, t, s].item()
                    for k in range(num_types):
                        intensity = baseline[k].item()
                        for j in range(num_types):
                            for t_j in event_times[b][j]:
                                if t_j <= current_time:
                                    delta = current_time - t_j
                                    intensity += adjacency[k, j].item() * np.exp(-decays[k, j].item() * delta)
                        intensities[b, t, s, k] = intensity
        return intensities
   

def ensure_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)