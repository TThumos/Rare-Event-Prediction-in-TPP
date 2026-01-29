import math
import random
import torch

def generate_synthetic_sequences(model, num_seqs, max_len, num_event_types, freq, max_time, 
                                 over_sample_rate=2.0, num_samples_for_bounds=100, device='cpu'):
    """
    Generate synthetic event sequences from a THP model via adaptive thinning.

    Args:
        model: a trained THP model (TorchBaseModel) with compute_intensities_at_sample_times(...)
        num_seqs: number of sequences to generate
        max_len: maximum number of events per sequence
        num_event_types: total number of event types
        max_time: maximum lookahead for one thinning step
        device: 'cpu' or 'cuda'
        over_sample_rate: factor to inflate the intensity upper bound
        num_samples_for_bounds: number of time points to sample for bounding intensity

    Returns:
        List of sequences (each a list of dicts with keys 
        {"time_since_start","time_since_last_event","type_event"}).
    """
    
    sequences = []
    for _ in range(num_seqs):
        # Initialize sequence with first event at time 0
        current_time = 10.0
        # Random first event type (0 to num_event_types-1)
        first_type = random.randrange(num_event_types)
        second_type = random.randrange(num_event_types)
        seq_times = [0.0, 10.0]               # absolute times of events
        seq_types = [first_type, second_type]    # model expects types
        seq_deltas = [0.0, 10.0]             # time since last (first is 0)
        
        # Start building the sequence
        for _ in range(1, max_len+49):
            # Compute an upper bound M on intensity over [0, max_time] after last event
            # Sample a grid of candidate offsets in [0, max_time]
            L = len(seq_times)
            dt_samples  = torch.rand(num_samples_for_bounds, device=device) * max_time
            sample_dt = torch.zeros((1, L, num_samples_for_bounds), device=device)             # [1, L, S]
            sample_dt[:, -1, :] = dt_samples
            # Prepare tensors for model input: shape [1, seq_len]
            t_seq = torch.tensor([seq_times], device=device)
            dt_seq = torch.tensor([seq_deltas], device=device)
            type_seq = torch.tensor([seq_types], dtype=torch.long, device=device)
            # Compute intensities at these sampled offsets (only last event state needed)
            with torch.no_grad():
                lambdas = model.compute_intensities_at_sample_times(
                    time_seqs=t_seq,
                    time_delta_seqs=dt_seq,
                    type_seqs=type_seq,
                    sample_dtimes=sample_dt,
                    compute_last_step_only=True
                )  # shape [1,1,num_samples,num_event_types]
            # Sum over event types to get total intensity at each sample point
            total_intensities = lambdas[0, 0, :, :].sum(dim=-1)  # shape [num_samples]
            # Determine upper bound M
            max_intensity_est = total_intensities.max().item()
            M = max_intensity_est * over_sample_rate
            # If bound is zero, no future events can occur
            if M <= 0:
                break
            
            # Now perform thinning: sample waiting time until next event
            accepted = False
            while not accepted:
                # Draw candidate dt from Exponential(M)
                u = random.random()
                dt = -math.log(u) / M
                new_time = current_time + dt
                # If beyond max_time horizon, give up and end sequence
                if dt > max_time:
                    continue
                # Compute intensity at this candidate time
                sample_dt_exact = torch.zeros((1, L, 1), device=device)
                sample_dt_exact[:, -1, 0] = float(dt)
                with torch.no_grad():
                    lambdas_exact = model.compute_intensities_at_sample_times(
                        time_seqs=t_seq,
                        time_delta_seqs=dt_seq,
                        type_seqs=type_seq,
                        sample_dtimes=sample_dt_exact,
                        compute_last_step_only=True
                    )  # shape [1,1,1,num_event_types]
                lambdas_exact = lambdas_exact[0, 0, 0, :]  # shape [num_event_types]
                total_intensity = lambdas_exact.sum().item()
                # If intensity is zero, no more events
                if total_intensity <= 0:
                    break
                # Accept or reject based on thinning
                if random.random() * M <= total_intensity:
                    # Accept this event time
                    accepted = True
                    delta_time = new_time - seq_times[-1]
                    # Sample event type proportional to intensity
                    # (we use the same lambdas_exact as weights)
                    if freq is None:
                        probs = lambdas_exact / lambdas_exact.sum()
                    else:
                        probs = freq
                    event_type = torch.multinomial(probs, 1).item()
                    # Record the event
                    seq_times.append(new_time)
                    seq_types.append(event_type)
                    seq_deltas.append(delta_time)
                    current_time = new_time
                else:
                    # Reject: move current time forward and continue
                    current_time = new_time
                    
            # Check if we exited without adding a new event (intensity 0 or horizon)
            if len(seq_times) <= len(seq_deltas) - 1:
                break  # stop generating this sequence
        
        # Convert to output format: list of dicts
        seq_times = seq_times[50:]
        seq_deltas = seq_deltas[50:]
        seq_types = seq_types[50:]
        seq_events = []
        last_time = 0.0
        for t, dt, tp in zip(seq_times, seq_deltas, seq_types):
            seq_events.append({
                "time_since_start": float(t),
                "time_since_last_event": float(dt),
                "type_event": int(tp)  # subtract 1 to return to 0..K-1 indexing
            })
            last_time = t
        sequences.append(seq_events)
    
    return sequences