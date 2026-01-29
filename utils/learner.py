import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def train_one_epoch(model, data_loader, optimizer, weight):
    model.train()

    for batch in data_loader:
        optimizer.zero_grad()
        bce_loss, num_rare, loglike_loss, num_events = model.loss(batch)
        loss = weight[0] * bce_loss/num_rare + weight[1] * loglike_loss/num_events
        loss.backward()
        optimizer.step()
    
    return loss


def plot_pr_curves(labels_list, probs_list, class_indices, title_prefix="PR Curves for Rare Classes", figsize=(15, 10)):

    n_classes = len(class_indices)
    
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    pos_ratios = [labels.mean() for labels in labels_list]
    
    for idx, (class_idx, labels_k, probs_k) in enumerate(zip(class_indices, labels_list, probs_list)):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        precision, recall, _ = precision_recall_curve(labels_k, probs_k)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, 'skyblue', linewidth=1, label=f'PR curve (PR-AUC = {pr_auc:.3f})')
        
        pos_ratio = pos_ratios[idx]
        ax.axhline(y=pos_ratio, color='orange', linestyle='--', linewidth=1, 
                   label=f'Random (AP = {pos_ratio:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'Type {class_idx} - PR-AUC: {pr_auc:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    total_plots = n_rows * n_cols
    if total_plots > n_classes:
        for idx in range(n_classes, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
    
    fig.suptitle(f'{title_prefix} (Total: {n_classes} classes)', fontsize=14, fontweight='bold', y=1.02)
    
    plt.show()


def max_f1_via_pr_curve_with_noise(
    probs_1d: np.ndarray,
    labels_1d: np.ndarray,
    n_repeat: int = 10,
    noise_std: float = 1e-6,
    seed: int = 42,
):
    """
    用 precision_recall_curve 的阈值序列来计算 max F1。
    对 probs 加噪声重复 n_repeat 次，返回 max F1 的均值/方差。
    """
    probs_1d = probs_1d.astype(np.float64)
    labels_1d = labels_1d.astype(np.int32)

    rng = np.random.default_rng(seed)
    max_f1_list = []

    eps = 1e-12
    for _ in range(n_repeat):
        noisy = probs_1d + rng.normal(0.0, noise_std, size=probs_1d.shape)
        noisy = np.clip(noisy, 0.0, 1.0)

        precision, recall, thresholds = precision_recall_curve(labels_1d, noisy)
        # precision/recall 比 thresholds 多 1 个点；与 thresholds 对齐用 [1:]
        f1 = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + eps)

        max_f1_list.append(float(np.max(f1)) if f1.size > 0 else 0.0)

    return float(np.mean(max_f1_list)), float(np.std(max_f1_list))


def evaluate(model, data_loader, alpha, weight, if_plot, compute_f1, n_repeat=10, noise_std=1e-6):
    model.eval()
    rare_class_mask = (model.freq > alpha[0]) & (model.freq < alpha[1])
    rare_indices = np.where(rare_class_mask.detach().cpu().numpy())[0]

    if len(rare_indices) == 0:
        print("No rare events under given alpha.")
        return None, None

    total_bce_loss = 0.0
    total_num_rare = 0
    total_loglike_loss = 0.0
    total_num_events = 0

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            bce_loss, num_rare, loglike_loss, num_events = model.loss(batch)
            total_bce_loss += bce_loss
            total_num_rare += num_rare
            total_loglike_loss += loglike_loss
            total_num_events += num_events

            if if_plot or compute_f1:
                probs, labels = model.forward(batch)  # [B, N, K]
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu().int())

    if (if_plot or compute_f1) and len(all_probs) > 0:
        all_probs = torch.cat(all_probs, dim=0).view(-1, model.freq.size(0))   # [E, K]
        all_labels = torch.cat(all_labels, dim=0).view(-1, model.freq.size(0)) # [E, K]

    # ---- 画 PR（你原逻辑保留）----
    if if_plot:
        rare_labels_list, rare_probs_list = [], []
        for k in rare_indices:
            rare_probs_list.append(all_probs[:, k].numpy())
            rare_labels_list.append(all_labels[:, k].numpy())

        plot_pr_curves(
            rare_labels_list,
            rare_probs_list,
            rare_indices,
            title_prefix=f"PR Curves for Rare Classes (α={alpha})",
            figsize=(15, 3 * ((len(rare_indices) + 2) // 3))
        )

    # ---- max F1（precision_recall_curve + 噪声重复）----
    f1_summary = None
    if compute_f1:
        per_class = {}
        per_class_means = []

        for k in rare_indices:
            probs_k = all_probs[:, k].numpy()
            labels_k = all_labels[:, k].numpy()

            mean_max_f1, std_max_f1 = max_f1_via_pr_curve_with_noise(
                probs_k, labels_k,
                n_repeat=n_repeat,
                noise_std=noise_std
            )
            per_class[int(k)] = {"max_f1_mean": mean_max_f1, "max_f1_std": std_max_f1}
            per_class_means.append(mean_max_f1)

        macro_mean = float(np.mean(per_class_means)) if len(per_class_means) else 0.0
        f1_summary = {"macro_max_f1_mean": macro_mean, "per_class": per_class}

        print(f"[F1] α={alpha}, noise_std={noise_std}, repeat={n_repeat}")
        print(f"[F1] Macro mean max-F1 over rare classes: {macro_mean:.6f}")
        for k in rare_indices:
            info = per_class[int(k)]
            print(f"  - class {int(k)}: max-F1(mean±std) = {info['max_f1_mean']:.6f} ± {info['max_f1_std']:.6f}")

    # ---- loss ----
    loss = weight[0] * total_bce_loss / total_num_rare + weight[1] * total_loglike_loss / total_num_events
    return loss, f1_summary