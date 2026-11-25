from pruning_methods.pruning_utils import find_layers
import torch 
from general_utils.config import config
from general_utils import utils
from pruning_methods.OATS.oats_wrapper import WrappedOATS
from models.model_utils import get_layer0_inputs
from tqdm import tqdm
from pruning_methods.pruning_utils import load_checkpoint, save_checkpoint
from pruning_methods.OATS.compressed_linear import CompressedLinear, CompressedQKV
from pruning_methods.OATS.oats_utils import _get_submodules
import os

@torch.no_grad()
def prune_oats_compress(model_adapter, sparsity, layerwise_sparsity_ratios, calib_loader, prune_hyperparams, checkpoint_path, prune_n=0, prune_m=0, *, quantization=None, report=None, projection_pool=None):

    if projection_pool is not None:
        if (prune_n, prune_m) != (2, 4) or any(len(layer.qkv_names) != 3 for layer in model_adapter.get_layers()):
            raise ValueError("Projection parallelism currently supports separate Llama projections with 2:4 sparsity")

    if quantization is not None:
        from pruning_methods.OATS.awq.quantize import AWQConfig, InLoopQuantizer
        if (prune_n, prune_m) != (2, 4) or any(len(layer.qkv_names) != 3 for layer in model_adapter.get_layers()):
            raise ValueError("OATS AWQ currently supports separate Llama projections with 2:4 sparsity")
        awq_config = AWQConfig(**quantization.get('config', {}))

    model_adapter.model.eval()
    use_cache = model_adapter.model.config.use_cache
    print(use_cache)
    
    model_adapter.model.config.use_cache = False 
    
    unscaled = prune_hyperparams['unscaled']
    rank_ratio = prune_hyperparams['rank_ratio']
    fixed_rank = prune_hyperparams.get('fixed_rank', None)
    prune_level = prune_hyperparams['prune_level']
    num_iters = prune_hyperparams['num_iters']

    inps, args, kwargs = [],  [], []
    prune_start_idx = -1

    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        _, pruned_args, prune_start_idx, layerwise_sparsity_ratios, kwargs = load_checkpoint(checkpoint_path +  "/prune_chkpt.pt")
        print("Resuming pruning from: " + str(prune_start_idx))
        print(config.dtype)
    else:
        for batch in calib_loader:
            inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
            args.append(args_batch)
            kwargs.append(kwargs_batch)
            inps.append(inp_batch)
        pruned_args = args
    
    layers = model_adapter.get_layers()
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Pruning using OATS")):
        if layer_idx > prune_start_idx:
            # ========== Setup Transformer Block Sparsity for OWL ================
            if layerwise_sparsity_ratios is not None:
                assert len(layerwise_sparsity_ratios) == len(layers)
                dense_alloc = 1 - layerwise_sparsity_ratios[layer_idx]
                print("Pruning Layer: " + str(layer_idx) + " using OWL to dense allocation: " + str(dense_alloc)  + ". OG dense allocation: " + str(1 - sparsity))
            else:
                dense_alloc = 1 - sparsity
            # ========== Setup hooks and wrap layers ==============================
            subset = find_layers(layer_adapter.layer)
            wrapped_layers = {}

            for name in subset:
                wrapped_layers[name] = WrappedOATS(
                    subset[name], activation_tokens=quantization.get('tokens', 512) if quantization else 0,
                    calibration_sequences=quantization.get('sequences', 128) if quantization else 128)
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            # =========== Precompute quantities =================================
            layer_adapter.layer.to(config.device)
            for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(pruned_args, kwargs)):
                layer_args_batch, layer_kwargs_batch = utils.map_tensors(
                    [layer_args_batch, layer_kwargs_batch], device=config.device
                )
                out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            for h in handles:
                h.remove()
            # ============== Prune the weights in the layer ==============================

            if len(layer_adapter.qkv_names) == 1:
                qkv_name = layer_adapter.qkv_names[0]

                diag_approx = wrapped_layers[qkv_name].scaler_row.clone().reshape((1,-1)).to(config.device)
                if unscaled:
                    diag_approx = torch.ones_like(diag_approx).float().to(config.device)

                qkv_weight = subset[qkv_name].weight.data.clone().detach().float()

                q_proj = qkv_weight[                                    : layer_adapter.get_qkv_partition()[0], : ].clone().detach().float()
                k_proj = qkv_weight[layer_adapter.get_qkv_partition()[0]: layer_adapter.get_qkv_partition()[1], : ].clone().detach().float()
                v_proj = qkv_weight[layer_adapter.get_qkv_partition()[1]: ,                                     : ].clone().detach().float()

                d_in = qkv_weight.shape[1]

                if fixed_rank is not None:
                    q_rank = max(1, min(int(fixed_rank), q_proj.shape[0], d_in))
                    k_rank = max(1, min(int(fixed_rank), k_proj.shape[0], d_in))
                    v_rank = max(1, min(int(fixed_rank), v_proj.shape[0], d_in))
                else:
                    q_rank = int(rank_ratio  * dense_alloc * (q_proj.shape[0]*d_in)/(q_proj.shape[0] + d_in))
                    k_rank = int(rank_ratio  * dense_alloc * (k_proj.shape[0]*d_in)/(k_proj.shape[0] + d_in))
                    v_rank = int(rank_ratio  * dense_alloc * (v_proj.shape[0]*d_in)/(v_proj.shape[0] + d_in))
                
                compressed_module = CompressedQKV(qkv_weight.shape[1],     \
                                                  q_rank, q_proj.shape[0], \
                                                  k_rank, k_proj.shape[0], \
                                                  v_rank, v_proj.shape[0], \
                                                  bias = subset[qkv_name].bias is not None, dtype=config.dtype) 
                
                for qkv_idx, qkv_mat in enumerate([q_proj, k_proj, v_proj]):
                    d_out = qkv_mat.shape[0]
                    d_in = qkv_mat.shape[1]

                    if prune_n != 0:
                        unstruct_sparse = prune_n/prune_m
                        if fixed_rank is not None:
                            target_rank = max(1, min(int(fixed_rank), d_out, d_in))
                            print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                            print("Target Rank for Low-Rank Term (fixed): " + str(target_rank))
                        else:
                            dense_alloc = unstruct_sparse/(1.0- rank_ratio)
                            target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                            print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                            print("Target Rank for Low-Rank Term: " + str(target_rank))
                            print("Dense Allocation for Layer when Doing N:M Sparsity: " + str(dense_alloc))
                    else:
                        if fixed_rank is not None:
                            raise NotImplementedError("fixed_rank is only implemented for structured N:M OATS in this patch.")
                        target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                        unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc

                    lrc_V, lrc_U, sparse_comp = altern_ls(qkv_mat, diag_approx, \
                                            num_iters, target_rank, unstruct_sparse, \
                                            prune_level = prune_level, prune_n=prune_n, prune_m=prune_m)
                    
                    if qkv_idx == 0:
                        compressed_module.q_V.data = lrc_V.clone().to(config.dtype)
                        compressed_module.q_U.data = lrc_U.clone().to(config.dtype)
                        compressed_module.q_S.data = sparse_comp.clone().to(config.dtype)
                    elif qkv_idx == 1:
                        compressed_module.k_V.data = lrc_V.clone().to(config.dtype)
                        compressed_module.k_U.data = lrc_U.clone().to(config.dtype)
                        compressed_module.k_S.data = sparse_comp.clone().to(config.dtype)
                    elif qkv_idx == 2:
                        compressed_module.v_V.data = lrc_V.clone().to(config.dtype)
                        compressed_module.v_U.data = lrc_U.clone().to(config.dtype)
                        compressed_module.v_S.data = sparse_comp.clone().to(config.dtype)
                    else:
                        raise ValueError
                
                if subset[qkv_name].bias is not None:
                    compressed_module.bias.data = subset[qkv_name].bias.data.detach().clone()
                
                parent, target, target_name = _get_submodules(layer_adapter.layer, qkv_name)
                setattr(parent, target_name, compressed_module)
                
                del subset[qkv_name]
            
            # Prune remaining weights in model
            parallel_tasks = []
            solve_device = config.device
            if projection_pool is not None:
                # All calibration statistics are fixed before dispatch. Workers only
                # receive immutable CPU snapshots, not the live transformer block.
                layer_adapter.layer.cpu()
                solve_device = torch.device('cpu')
            for name in subset:
                print(f"pruning layer {layer_idx} name {name}")
                diag_approx = wrapped_layers[name].scaler_row.clone().reshape((1,-1)).to(solve_device)
                if unscaled:
                    diag_approx = torch.ones_like(diag_approx).float().to(solve_device)
                
                orig_weight = subset[name].weight.data.clone().detach().float().to(solve_device)
                d_out = orig_weight.shape[0]
                d_in = orig_weight.shape[1]

                
                if prune_n != 0 :
                    unstruct_sparse = prune_n/prune_m
                    if fixed_rank is not None:
                        target_rank = max(1, min(int(fixed_rank), d_out, d_in))
                        print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                        print("Target Rank for Low-Rank Term (fixed): " + str(target_rank))
                    else:
                        dense_alloc = unstruct_sparse/(1.0- rank_ratio)
                        target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                        print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                        print("Target Rank for Low-Rank Term: " + str(target_rank))
                        print("Dense Allocation for Layer when Doing N:M Sparsity: " + str(dense_alloc))
                else:
                    if fixed_rank is not None:
                        raise NotImplementedError("fixed_rank is only implemented for structured N:M OATS in this patch.")
                    target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                    unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc
                
                if projection_pool is not None:
                    parallel_tasks.append({
                        'name': name, 'weight': orig_weight, 'diagonal': diag_approx,
                        'samples': torch.cat(wrapped_layers[name].activation_samples) if quantization is not None else None,
                        'quantization': quantization, 'dtype': config.dtype,
                        'num_iters': num_iters, 'rank': target_rank, 'sparsity': unstruct_sparse,
                        'prune_level': prune_level, 'prune_n': prune_n, 'prune_m': prune_m,
                        'report': report is not None,
                    })
                    continue

                quantizer = None
                if quantization is not None:
                    samples = torch.cat(wrapped_layers[name].activation_samples).to(config.device)
                    quantizer = InLoopQuantizer(samples, awq_config, config.dtype)

                def module_report(record):
                    if report is not None:
                        report({'layer': layer_idx, 'module': name, **record})

                lrc_V, lrc_U, sparse_comp = altern_ls(orig_weight, diag_approx,\
                                        num_iters, target_rank, unstruct_sparse,\
                                        prune_level = prune_level, prune_n=prune_n, prune_m=prune_m,
                                        quantizer=quantizer, report=module_report if report is not None else None)
                del quantizer
                
                
                # replace module
                replace_linear(name, subset, layer_adapter, d_in, target_rank, d_out, lrc_V, lrc_U, sparse_comp)

            if projection_pool is not None:
                print(f"[projection parallel] layer {layer_idx}: {len(parallel_tasks)} tasks on {projection_pool.devices}", flush=True)
                solved = projection_pool.solve_layer(parallel_tasks)
                for result in solved:
                    name = result['name']
                    lrc_V, lrc_U, sparse_comp = result['factors']
                    d_out, d_in = subset[name].weight.shape
                    replace_linear(name, subset, layer_adapter, d_in, lrc_V.shape[0], d_out,
                                   lrc_V, lrc_U, sparse_comp)
                    if report is not None:
                        for record in result['records']:
                            report({'layer': layer_idx, 'module': name,
                                    'worker_device': result['device'], **record})
                del solved, parallel_tasks
            # ============== Recalculate outputs with pruned weight ====================
            pruned_outs = []

            layer_adapter.layer.to(config.device)
            for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(pruned_args, kwargs)):
                layer_args_batch, layer_kwargs_batch = utils.map_tensors(
                    [layer_args_batch, layer_kwargs_batch], device=config.device
                )
                out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
                if isinstance(out, tuple):
                    out = out[layer_adapter.hidden_states_output_position]
                out = out.cpu()
                pruned_outs.append(out)
            
            for batch_idx, pruned_out in enumerate(pruned_outs):
                pruned_args[batch_idx] = layer_adapter.get_updated_args(
                    pruned_out.cpu(),
                    pruned_args[batch_idx],
                )
            
            layer_adapter.layer.to('cpu')

            torch.save(layer_adapter.layer.state_dict(), checkpoint_path + "/oats_chkpt_" + str(layer_idx) + ".pt")
            save_checkpoint(None, pruned_args, layer_idx, layerwise_sparsity_ratios, kwargs, checkpoint_path + "/prune_chkpt.pt")
            model_adapter.model.config.use_cache = False

            # Run GC and cleanup GPU memory
            utils.cleanup_memory()

    model_adapter.model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def replace_linear(name, subset, layer_adapter, d_in, target_rank, d_out, V, U, sparse_comp):
    parent, target, target_name = _get_submodules(layer_adapter.layer, name)
    new_module = CompressedLinear(d_in, target_rank, d_out, bias=subset[name].bias is not None, dtype=config.dtype)
    new_module.V.data = V.clone().to(config.dtype)
    new_module.U.data = U.clone().to(config.dtype)
    new_module.S.data = sparse_comp.clone().to(config.dtype)
    if subset[name].bias is not None:
        new_module.bias.data = subset[name].bias.data.detach().clone()
    setattr(parent, target_name, new_module)

def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_level = "row", prune_n=0, prune_m=0, *, quantizer=None, report=None, observer=None):
    if quantizer is not None:
        from pruning_methods.OATS.awq.quantize import validate_tensor
        validate_tensor(weight, "OATS weight", ndim=2)
        if num_iters < 1 or not 0 < target_rank <= min(weight.shape):
            raise ValueError("Positive iterations and a valid rank are required")
        if (prune_n, prune_m) != (2, 4) or weight.shape[1] % 4:
            raise ValueError("AWQ requires input width divisible by 4 and 2:4 sparsity")
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
        if quantizer is not None:
            # Keep OATS's SVD and factor convention. Quantize in original coordinates.
            lrc_V = V[:target_rank, :].detach().clone() * (1 / sqrt_diag)
            lrc_U = (U[:, :target_rank] @ torch.diag(S)[:target_rank, :target_rank]).detach().clone()
            lrc_V, lrc_U = quantizer.pair(
                lrc_V, lrc_U, weight - sparse_component / sqrt_diag, iter_idx)
            low_rank_component = (lrc_U @ lrc_V) * sqrt_diag
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
        if quantizer is not None:
            sparse_q = quantizer.sparse(
                sparse_component / sqrt_diag, weight - lrc_U @ lrc_V, iter_idx)
            sparse_component = sparse_q * sqrt_diag

        if report is not None or observer is not None:
            current_V = lrc_V if quantizer is not None else V[:target_rank, :].detach().clone() / sqrt_diag
            current_U = lrc_U if quantizer is not None else (U[:, :target_rank] @ torch.diag(S)[:target_rank, :target_rank]).detach().clone()
            current_S = sparse_q if quantizer is not None else sparse_component / sqrt_diag
            if observer is not None:
                observer(iter_idx + 1, current_V, current_U, current_S)
            if report is not None:
                residual = weight - current_S - current_U @ current_V
                record = {
                    'iteration': iter_idx + 1, 'returned_iteration': num_iters,
                    'iterate_selection': 'last',
                    'rel_frob_err': float(residual.norm() / weight.norm().clamp_min(1e-12)),
                    'weighted_rel_frob_err': float((residual * sqrt_diag).norm() / scaled_weight.norm().clamp_min(1e-12)),
                }
                if quantizer is not None:
                    record.update(getattr(quantizer, 'info', {}))
                    if hasattr(quantizer, 'x'):
                        x = quantizer.x
                        predicted = torch.nn.functional.linear(x, current_S.to(x.dtype)) + torch.nn.functional.linear(torch.nn.functional.linear(x, current_V.to(x.dtype)), current_U.to(x.dtype))
                        reference = torch.nn.functional.linear(x.float(), weight.float())
                        record['output_mse'] = float((predicted.float() - reference).square().mean())
                report(record)
        
    
    if quantizer is not None:
        # Return the last effective quantized factors, never re-SVD or select a best iterate.
        return lrc_V, lrc_U, sparse_q

    # low_rank_compressed_V = (V[:target_rank, :]).detach().clone() * (1/torch.sqrt(diag_approx))
    low_rank_compressed_V = (V[:target_rank, :]).detach().clone() * (1 / sqrt_diag)
    low_rank_compressed_U = (U[:, :target_rank] @ torch.diag(S)[:target_rank, : target_rank]).detach().clone()
    # sparse_comp = sparse_component * (1/torch.sqrt(diag_approx))
    sparse_comp = sparse_component * (1 / sqrt_diag)
    return low_rank_compressed_V, low_rank_compressed_U, sparse_comp
