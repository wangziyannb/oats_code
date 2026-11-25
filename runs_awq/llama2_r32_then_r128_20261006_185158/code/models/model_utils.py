from typing import Any

import torch
from torch import Tensor

from general_utils import utils
from general_utils.config import config
from models.model_adapter import ModelAdapter


def get_layer0_inputs(model_adapter: ModelAdapter, batch: Tensor) -> tuple[Tensor, tuple, dict[str, Any]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for embed_module in model_adapter.get_embeddings():
        embed_module.to(config.device)


    class Catcher(torch.nn.Module):
        def __init__(self, orig_layer=None):
            super().__init__()
            if orig_layer is not None and hasattr(orig_layer, "attention_type"):
                self.attention_type = orig_layer.attention_type

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    orig_layer0 = layer0_adapter.layer

    layer0_catcher = Catcher(orig_layer0)
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        model_batch = {k: v for k, v in batch.items() if k != "labels"}
        model_batch = utils.map_tensors(model_batch, device=config.device)
        model_adapter.model(**model_batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for embed_module in model_adapter.get_embeddings():
        embed_module.to('cpu')

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs
