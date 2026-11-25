import torch
from pruning_methods.pruning_utils import find_layers
import os
from pruning_methods.pruning_utils import load_checkpoint, save_checkpoint
from general_utils.config import config
from general_utils import utils
from models.model_utils import get_layer0_inputs
import torch.nn as nn
import transformers
import time
from tqdm import tqdm
import pathlib
import shutil
from pruning_methods.DSnoT.dsnot_wrapper import DSnotWrapper

@torch.no_grad()
def prune_DSnoT(
    model_adapter, tokenizer, sparsity, layerwise_sparsity_ratios, calib_loader, prune_hyperparams, checkpoint_path, prune_n=0, prune_m=0
):
    model_adapter.model.eval()
    use_cache = model_adapter.model.config.use_cache
    model_adapter.model.config.use_cache = False

    max_cycle_time = prune_hyperparams['max_cycle_time']
    pow_of_var_regrowing = prune_hyperparams['pow_of_var_regrowing']
    update_threshold = prune_hyperparams['update_threshold']
    without_same_sign =  prune_hyperparams['without_same_sign']
    initial_method = prune_hyperparams['initial_method']

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
    
    total_time = 0
    layers = model_adapter.get_layers()

    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Pruning using DSNOT")):
        if layer_idx > prune_start_idx:
            # ========== Setup Transformer Block Sparsity for OWL ================
            if layerwise_sparsity_ratios is not None:
                assert len(layerwise_sparsity_ratios) == len(layers)
                layer_sparsity = layerwise_sparsity_ratios[layer_idx]
                print("Pruning Layer: " + str(layer_idx) + " using OWL to sparsity: " + str(layer_sparsity)  + ". OG sparsity: " + str(sparsity))
            else:
                layer_sparsity = sparsity
            
            # ========== Setup hooks and wrap layers ==============================
            subset = find_layers(layer_adapter.layer)
            wrapped_layers = {}

            for name in subset:
                wrapped_layers[name] = DSnotWrapper(
                    subset[name],
                    initial_method=initial_method
                )
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            # =========== Feed input through layers =================================
            layer_adapter.layer.to(config.device)
            for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(pruned_args, kwargs)):
                layer_args_batch, layer_kwargs_batch = utils.map_tensors(
                    [layer_args_batch, layer_kwargs_batch], device=config.device
                )
                out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)

            for h in handles:
                h.remove()

            # ============== Prune the weights in the layer ==============================
            for name in subset:
                print(f"pruning layer {layer_idx} name {name}")
                start_time = time.time()

                DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1)).to(config.device)

                if initial_method == "wanda":
                    initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)).to(config.device)
                    )
                elif initial_method == "magnitude":
                    initial_metric = torch.abs(subset[name].weight.data)
                elif initial_method == "sparsegpt":
                    W = subset[name].weight.data.clone()
                    if isinstance(subset[name], nn.Conv2d):
                        W = W.flatten(1)
                    if isinstance(subset[name], transformers.Conv1D):
                        W = W.t()
                    W = W.float()

                    H = wrapped_layers[name].H
                    # del wrapped_layers[name].H
                    dead = torch.diag(H) == 0
                    H[dead, dead] = 1
                    W[:, dead] = 0

                    percdamp = 0.01
                    damp = percdamp * torch.mean(torch.diag(H))
                    diag = torch.arange(
                        wrapped_layers[name].columns, device=wrapped_layers[name].dev
                    )
                    H[diag, diag] += damp
                    H = torch.linalg.cholesky(H)
                    H = torch.cholesky_inverse(H)
                    H = torch.linalg.cholesky(H, upper=True)
                    Hinv = H

                    initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1))).to(config.device) ** 2

                    W = None

                weight_mask = torch.zeros_like(initial_metric) == 1

                if prune_n != 0:
                    initial_prune_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)
                    initial_res_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)

                    for ii in range(initial_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + prune_m)].float()
                            _, tmp_all_indices = torch.sort(tmp, dim=1)
                            tmp_all_indices += ii
                            res_prune_n = prune_m - prune_n
                            tmp_indices, tmp_res_indices = torch.split(
                                tmp_all_indices,
                                split_size_or_sections=[prune_n, res_prune_n],
                                dim=1,
                            )

                            initial_prune_indices = torch.cat(
                                (initial_prune_indices, tmp_indices), dim=1
                            )
                            initial_res_indices = torch.cat(
                                (initial_res_indices, tmp_res_indices), dim=1
                            )
                            weight_mask.scatter_(1, tmp_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)

                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)).to(config.device),
                            pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one
                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    initial_metric.scatter_(1, initial_prune_indices, float("inf"))
                    W_metric_max_value = (torch.max(initial_metric, dim=1, keepdim=True)[0] + 1)

                    cycle_time = 1
                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    while not (
                        torch.all(update_mask == False)
                        or cycle_time > max_cycle_time
                    ):
                        cycle_time += 1

                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )
                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        recover_block_start_indice = (
                            regrowing_indice - regrowing_indice % prune_m
                        )

                        recover_block_indices = (
                            torch.arange(
                                0, prune_m, device=recover_block_start_indice.device
                            ).repeat(recover_block_start_indice.shape[1], 1)
                            + recover_block_start_indice
                        )

                        pruning_block = torch.gather(
                            initial_metric, 1, recover_block_indices.to(torch.int64)
                        )

                        pruning_wanda_metric, pruning_indice = torch.topk(
                            pruning_block, 1, dim=1, largest=False
                        )

                        pruning_indice += recover_block_start_indice

                        
                        pruning_metric = DSnoT_metric.gather( 1, pruning_indice.to(torch.int64) )
                        

                        reconstruction_error_after = ( reconstruction_error + pruning_metric - regrowing_metric )

                        update_mask = (update_mask & ( initialize_error_sign == torch.sign(reconstruction_error_after) ) & ( abs(reconstruction_error) > update_threshold))

                        initial_metric.scatter_(1, pruning_indice, W_metric_max_value)

                        weight_mask.scatter_(1, pruning_indice, update_mask)

                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )
                else:
                    _, sorted_initial_indice = torch.sort(
                        initial_metric, dim=-1, stable=True
                    )

                    sparsity_num = int(initial_metric.shape[1] * layer_sparsity)
                    res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                    initial_prune_indices, initial_res_indices = torch.split(
                        sorted_initial_indice,
                        split_size_or_sections=[sparsity_num, res_sparsity_num],
                        dim=1,
                    )

                    weight_mask.scatter_(1, initial_prune_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)).to(config.device)
                    )

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)
                    reconstruction_error = torch.sum(
                        metric_for_regrowing, dim=1, keepdim=True
                    )
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)).to(config.device),
                            pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(
                        metric_for_regrowing, dim=1, stable=True
                    )

                    wanda_metric.scatter_(1, initial_prune_indices, float("inf"))
                    wanda_res_indices, _ = torch.split(
                        torch.sort(wanda_metric, dim=1, stable=True)[1],
                        split_size_or_sections=[res_sparsity_num, sparsity_num],
                        dim=1,
                    )
                    reorder_indice_of_pruning_indice = return_reorder_indice(
                        torch.gather(DSnoT_metric, 1, wanda_res_indices)
                    )
                    pruning_indices_block = torch.gather(
                        wanda_res_indices, 1, reorder_indice_of_pruning_indice
                    )

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one

                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    indice_indice_list_for_pruning = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = pruning_indices_block.shape[-1] - 1
                    indice_indice_list_for_pruning[:, 1] = last_one

                    update_num_for_pruning = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_pruning[:, 1] = -1

                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    cycle_time = 0
                    while not ( torch.all(update_mask == False) or cycle_time >= max_cycle_time ):
                        cycle_time += 1
                        
                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )

                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )

                        # pruning
                        indice_of_indice_indice_list_for_pruning = (
                            (reconstruction_error < 0).int().to(torch.int64)
                        )

                        indice_indice_for_pruning = torch.gather(
                            indice_indice_list_for_pruning,
                            1,
                            indice_of_indice_indice_list_for_pruning,
                        )

                        pruning_indice = torch.gather(
                            pruning_indices_block,
                            1,
                            indice_indice_for_pruning.to(torch.int64),
                        )

                        pruning_metric = DSnoT_metric.gather(
                            1, pruning_indice.to(torch.int64)
                        )

                        indice_indice_list_for_pruning.scatter_(
                            1,
                            indice_of_indice_indice_list_for_pruning, 
                            indice_indice_for_pruning
                            + update_num_for_pruning.gather(
                                1, indice_of_indice_indice_list_for_pruning
                            ),
                        )

                        # change mask
                        reconstruction_error_after = (
                            reconstruction_error + pruning_metric - regrowing_metric
                        )

                        if without_same_sign:
                            update_mask = update_mask & (
                                abs(reconstruction_error) > update_threshold
                            )
                        else:
                            update_mask = (
                                update_mask
                                & (abs(reconstruction_error) > update_threshold)
                                & (
                                    initialize_error_sign
                                    == torch.sign(reconstruction_error_after)
                                )
                            )

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                
                subset[name].weight.data[weight_mask] = 0

                end_time = time.time()
                total_time += end_time - start_time
            
            DSnoT_metric = None
            initial_metric = None
            wanda_metric = None
            metric_for_regrowing = None
            utils.cleanup_memory()

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

            if layer_idx % 2 == 1:
                tmp_chkpt_path = checkpoint_path + "_tmp"
                pathlib.Path(tmp_chkpt_path).mkdir(parents=True, exist_ok=True)

                model_adapter.model.config.use_cache = use_cache
                model_adapter.model.save_pretrained(tmp_chkpt_path)
                tokenizer.save_pretrained(tmp_chkpt_path)
                save_checkpoint(None, pruned_args, layer_idx, layerwise_sparsity_ratios, kwargs, tmp_chkpt_path + "/prune_chkpt.pt")

                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                
                os.replace(tmp_chkpt_path, checkpoint_path)
                model_adapter.model.config.use_cache = False 
            
            # Run GC and cleanup GPU memory
            utils.cleanup_memory()

    model_adapter.model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice