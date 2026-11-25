"""Frozen pre-AWQ altern_ls regression reference. SHA256 of source: e47c63a5135d9462e49480b2ea14fca41f7a7d5c0c618d82e6314f32a54a376d"""
import torch
from general_utils.config import config

def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_level = "row", prune_n=0, prune_m=0):
    if not torch.isfinite(diag_approx).all():
        bad_mask = ~torch.isfinite(diag_approx)
        diag_safe = torch.nan_to_num(diag_approx.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        print(
            "diag_approx has non-finite values. ",
            f"bad={int(bad_mask.sum().item())}, ",
            f"min={float(diag_safe.min().item())}, ",
            f"max={float(diag_safe.max().item())}"
        )
        raise ValueError("diag_approx contains NaN/Inf before altern_ls")

    eps = 1e-8
    diag_approx = diag_approx.clamp_min(eps)
    sqrt_diag = torch.sqrt(diag_approx)
    scaled_weight = weight * sqrt_diag  # d_out x d_in

    # if diag_approx.isnan().any():
    #     print("Outliers have NaN. Exiting!")
    #     raise ValueError

    # scaled_weight = weight * torch.sqrt(diag_approx) # d_out x d_in
    sparse_component = torch.zeros_like(scaled_weight).to(config.device)
    for iter_idx in range(num_iters): 
        # Apply PCA
        U, S, V = torch.linalg.svd(scaled_weight - sparse_component , full_matrices=False)
        S[target_rank:] = 0
        low_rank_component = U @ torch.diag(S) @ V
        sparse_component = scaled_weight - low_rank_component

        # Prune the weight
        W_metric = sparse_component.clone()
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        
        if prune_n != 0:
            print("Applying N:M Sparsity")
            # structured n:m sparsity
            W_metric = torch.abs(W_metric)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_m - prune_n,dim=1, largest=False)[1], True)
        elif prune_level == "row":
            sort_res = torch.sort(torch.abs(W_metric), dim=-1, stable=True)
            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]* unstruct_sparse)]
            W_mask.scatter_(1, indices, True)
        elif prune_level == "global":
            sort_res = torch.sort(torch.flatten(torch.abs(W_metric)), stable=True)
            indices = sort_res[1][:int(W_metric.numel()* unstruct_sparse)]

            W_mask = torch.flatten(W_mask)
            W_mask[indices] = True
            W_mask = torch.unflatten(W_mask, 0 , (W_metric.shape[0], W_metric.shape[1]))
        else:
            assert ValueError
        sparse_component[W_mask] = 0
        
    
    # low_rank_compressed_V = (V[:target_rank, :]).detach().clone() * (1/torch.sqrt(diag_approx))
    low_rank_compressed_V = (V[:target_rank, :]).detach().clone() * (1 / sqrt_diag)
    low_rank_compressed_U = (U[:, :target_rank] @ torch.diag(S)[:target_rank, : target_rank]).detach().clone()
    # sparse_comp = sparse_component * (1/torch.sqrt(diag_approx))
    sparse_comp = sparse_component * (1 / sqrt_diag)
    return low_rank_compressed_V, low_rank_compressed_U, sparse_comp
