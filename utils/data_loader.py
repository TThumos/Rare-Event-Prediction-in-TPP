import torch
from torch.utils.data import Dataset, DataLoader


class TripleDatasetWithOptionalExtra(Dataset):
    def __init__(self, ds_origin, ds_kept=None, ds_residual=None, extra_tensor=None):
        n = len(ds_origin)

        if ds_kept is not None:
            assert len(ds_kept) == n
        if ds_residual is not None:
            assert len(ds_residual) == n
        if extra_tensor is not None:
            assert extra_tensor.size(0) == n

        self.ds_origin = ds_origin
        self.ds_kept = ds_kept
        self.ds_residual = ds_residual
        self.extra = extra_tensor

    def __len__(self):
        return len(self.ds_origin)

    def __getitem__(self, idx):
        o = self.ds_origin[idx]
        k = None if self.ds_kept is None else self.ds_kept[idx]
        r = None if self.ds_residual is None else self.ds_residual[idx]
        ex = None if self.extra is None else self.extra[idx]
        return o, k, r, ex


def build_triple_loader_with_optional_extra(loader_origin, loader_kept=None, loader_residual=None, extra_tensor=None):
    ds = TripleDatasetWithOptionalExtra(
        loader_origin.dataset,
        None if loader_kept is None else loader_kept.dataset,
        None if loader_residual is None else loader_residual.dataset,
        extra_tensor
    )

    collate_o = loader_origin.collate_fn
    collate_k = None if loader_kept is None else loader_kept.collate_fn
    collate_r = None if loader_residual is None else loader_residual.collate_fn

    def collate_fn(batch):
        o_list, k_list, r_list, ex_list = zip(*batch)

        origin_batch = collate_o(list(o_list))
        kept_batch = None if collate_k is None else collate_k(list(k_list))
        residual_batch = None if collate_r is None else collate_r(list(r_list))

        if extra_tensor is None:
            extra_batch = None
        else:
            extra_batch = torch.stack(
                [x if torch.is_tensor(x) else torch.as_tensor(x) for x in ex_list],
                dim=0
            )

        return {"origin": origin_batch, "kept": kept_batch, "residual": residual_batch, "extra": extra_batch}

    return DataLoader(
        ds,
        batch_size=loader_origin.batch_size,
        shuffle=False,
        num_workers=loader_origin.num_workers,
        pin_memory=getattr(loader_origin, "pin_memory", False),
        drop_last=getattr(loader_origin, "drop_last", False),
        collate_fn=collate_fn,
    )
