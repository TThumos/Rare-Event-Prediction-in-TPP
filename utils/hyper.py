import torch
from utils.poisson import PoissonProcess


class Hyper:

    def __init__(self, model1, model2, weight):
        
        self.model1 = model1
        self.model2 = model2
        self.weight = weight

    @torch.no_grad()
    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,       # [B, L]
        time_delta_seqs: torch.Tensor, # [B, L]
        type_seqs: torch.Tensor,       # [B, L]
        sample_dtimes: torch.Tensor,   # [B, L, S]
        compute_last_step_only: bool = False,
        **kwargs
    ) -> torch.Tensor:
        
        lambdas1 = self.model1.compute_intensities_at_sample_times(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            sample_dtimes=sample_dtimes,
            compute_last_step_only=compute_last_step_only,
        )

        lambdas2 = self.model2.compute_intensities_at_sample_times(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            sample_dtimes=sample_dtimes,
            compute_last_step_only=compute_last_step_only,
        )

        return self.weight[0] * lambdas1 + self.weight[1] * lambdas2
