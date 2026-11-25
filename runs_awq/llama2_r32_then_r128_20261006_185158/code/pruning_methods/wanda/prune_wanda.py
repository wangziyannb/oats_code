from pruning_methods.pruning_utils import find_layers
import torch 
from general_utils.config import config
from general_utils import utils
from pruning_methods.wanda.layerwrapper import WrappedGPT
from models.model_utils import get_layer0_inputs
from tqdm import tqdm
from pruning_methods.pruning_utils import load_checkpoint, save_checkpoint
import os
import time
import shutil
import pathlib

@torch.no_grad()
def prune_wanda(model_adapter, tokenizer, sparsity, layerwise_sparsity_ratios, calib_loader, checkpoint_path, prune_n=0, prune_m=0):

    model_adapter.model.eval()
    use_cache = model_adapter.model.config.use_cache 
    model_adapter.model.config.use_cache = False 
    

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
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Pruning using Wanda")):
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
                wrapped_layers[name] = WrappedGPT(subset[name])
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
                print(f"pruning layer {layer_idx} name {name}", flush=True)
                start_time = time.time()
                W_metric = torch.abs(subset[name].weight.data.clone().detach().cpu()) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]* layer_sparsity)]
                    W_mask.scatter_(1, indices, True)
                
                W_mask.to(config.device)
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for WANDA: {elapsed_time} seconds", flush=True)
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