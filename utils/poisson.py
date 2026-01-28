import torch
from typing import Optional, Union

class PoissonProcess:
    """
    Homogeneous multi-type Poisson process model with constant intensities.

    This class is compatible with generate_synthetic_sequences(...) because it provides
    compute_intensities_at_sample_times(...) returning shape [B, 1, S, K] when
    compute_last_step_only=True.
    """

    def __init__(
        self,
        num_event_types: int,
        rate: Union[float, torch.Tensor]
    ):
        """
        Parameters:
            num_event_types (int): Number of event types K.
            rate (float or Tensor): Either
                - scalar total rate (will be split uniformly across K), or
                - vector of per-type rates of shape [K], or
                - matrix of shape [B, K] (batch-specific; optional advanced usage).
        """
        if num_event_types <= 0:
            raise ValueError(f"num_event_types must be positive, got {num_event_types}")
        self.num_event_types = int(num_event_types)

        self._set_rate(rate)

    def _set_rate(self, rate: Union[float, torch.Tensor]) -> None:
        if isinstance(rate, (float, int)):
            # scalar total rate -> uniform per-type
            total = float(rate)
            if total <= 0:
                raise ValueError(f"rate must be > 0, got {total}")
            per_type = torch.full((self.num_event_types,), total / self.num_event_types)
            self.rate = per_type
        elif isinstance(rate, torch.Tensor):
            if rate.ndim == 1:
                if rate.numel() != self.num_event_types:
                    raise ValueError(f"rate shape must be [K], got {tuple(rate.shape)}")
                if torch.any(rate <= 0):
                    raise ValueError("All per-type rates must be > 0.")
                self.rate = rate
            elif rate.ndim == 2:
                # [B, K] batch-specific rates
                if rate.size(1) != self.num_event_types:
                    raise ValueError(f"rate shape must be [B,K] with K={self.num_event_types}, got {tuple(rate.shape)}")
                if torch.any(rate <= 0):
                    raise ValueError("All per-type rates must be > 0.")
                self.rate = rate
            else:
                raise ValueError(f"Unsupported rate tensor ndim={rate.ndim}")
        else:
            raise TypeError(f"Unsupported rate type: {type(rate)}")

    @torch.no_grad()
    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,       # [B, L]
        time_delta_seqs: torch.Tensor, # [B, L] (unused)
        type_seqs: torch.Tensor,       # [B, L] (unused)
        sample_dtimes: torch.Tensor,   # [B, L, S]
        compute_last_step_only: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Returns:
            lambdas: [B, 1, S, K] if compute_last_step_only=True else [B, L, S, K]
        """
        # Use the input tensor device to stay consistent with your generator
        B, L = time_seqs.shape
        S = sample_dtimes.size(2)
        K = self.num_event_types

        # Resolve per-batch per-type rates
        if self.rate.ndim == 1:
            # [K] -> [B, K]
            rate_bk = self.rate.unsqueeze(0).expand(B, -1)
        else:
            # [B, K]
            rate_bk = self.rate
            if rate_bk.size(0) != B:
                # if mismatch, broadcast first row
                rate_bk = rate_bk[:1, :].expand(B, -1)

        if compute_last_step_only:
            # [B, 1, S, K]
            out = rate_bk[:, None, None, :].expand(B, 1, S, K).contiguous()
        else:
            # [B, L, S, K]
            out = rate_bk[:, None, None, :].expand(B, L, S, K).contiguous()

        return out