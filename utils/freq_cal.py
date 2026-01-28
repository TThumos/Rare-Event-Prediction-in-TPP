import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_freq(loader, num_event_types):

    counts = torch.zeros(num_event_types)
    total_events = 0
    
    for batch in loader:
        type_seqs = batch['type_seqs']
        valid_mask = type_seqs != num_event_types
        valid_types = type_seqs[valid_mask]
        batch_counts = torch.bincount(valid_types, minlength=num_event_types)
        counts += batch_counts
        total_events += valid_types.numel()
    
    return counts, total_events


def plot_bar_chart(tensor, figsize=(10, 5)):
    
    k = len(tensor)
    
    values = tensor.cpu().numpy() if torch.is_tensor(tensor) else np.array(tensor)
    indices = np.arange(k)
    
    plt.figure(figsize=figsize)
    
    bars = plt.bar(indices, values, alpha=0.8)
    
    plt.xticks(indices)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    if k <= 30:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{values[i]:.5f}' if values[i] % 1 != 0 else f'{int(values[i])}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=7)
    
    plt.tight_layout()
    plt.show()