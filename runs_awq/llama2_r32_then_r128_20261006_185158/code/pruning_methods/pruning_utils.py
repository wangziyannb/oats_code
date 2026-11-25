import torch.nn as nn 
import os
import torch
from pruning_methods.wanda.layerwrapper import WrappedGPT
from tqdm import tqdm
from general_utils.config import config
from general_utils import utils
from models.model_utils import get_layer0_inputs
import numpy as np

@torch.no_grad()
def calc_outlier_ratio(model_adapter, sparsity, calib_loader, hyper_m=5, lamda=0.08, checkpoint_path=None):

    model_adapter.model.eval()
    use_cache = model_adapter.model.config.use_cache
    model_adapter.model.config.use_cache = False 

    inps, args, kwargs = [],  [], []

    for batch in calib_loader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        inps.append(inp_batch)
    
    all_layer_ratio=[]
    
    layers = model_adapter.get_layers()
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Detect outliers for OWL")):
        
        layer_wmetric = []

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
        # =========== Precompute quantities =================================
        outs = []
        layer_adapter.layer.to(config.device)

        for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(args, kwargs)):
            print(batch_idx)
            layer_args_batch, layer_kwargs_batch = utils.map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outs.append(out)

        for h in handles:
            h.remove()

        for batch_idx, out in enumerate(outs):
            args[batch_idx] = layer_adapter.get_updated_args(
                out.cpu(),
                args[batch_idx],
            )

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))).to(config.device)
            layer_wmetric.append(W_metric.cpu().clone())
        
        layer_adapter.layer.to('cpu')

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        for out_ratio in [hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)
    
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * lamda*2))
    all_layer_ratio= sparsity - (all_layer_ratio-np.mean(all_layer_ratio))

    model_adapter.model.config.use_cache = use_cache
    
    return all_layer_ratio

@torch.no_grad()
def check_outlier_mean(mask,threshold):

    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio=float(count)/total_params*100

    return outlier_ratio


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def save_checkpoint(orig_args, pruned_args, prune_start_idx, layerwise_sparsity_ratios, kwargs, checkpoint_path):

    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")
    training_state = {
        'orig_args': orig_args,
        'pruned_args': pruned_args,
        'prune_start_idx': prune_start_idx,
        'layerwise_sparsity_ratios': layerwise_sparsity_ratios,
        'kwargs': kwargs,
    }
    torch.save(training_state, temp_path)
    os.replace(temp_path, checkpoint_path)

def load_checkpoint(checkpoint_path, map_location=None):

    prune_state = torch.load(checkpoint_path, map_location=map_location)
    orig_args = prune_state['orig_args']
    pruned_args = prune_state['pruned_args']
    prune_start_idx = prune_state['prune_start_idx']
    kwargs = prune_state['kwargs']
    if 'layerwise_sparsity_ratios' in prune_state:
        layerwise_sparsity_ratios = prune_state['layerwise_sparsity_ratios']
    else:
        layerwise_sparsity_ratios = None
    
    return orig_args, pruned_args, prune_start_idx, layerwise_sparsity_ratios, kwargs


