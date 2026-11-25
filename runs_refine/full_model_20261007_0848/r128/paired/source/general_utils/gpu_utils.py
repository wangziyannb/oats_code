import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from general_utils import utils
from models.model_adapter import ModelAdapter


def distribute_model(model_adapter: ModelAdapter) -> None:
    """Distribute the model across available GPUs."""
    model = model_adapter.model
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model_adapter.no_split_module_classes,
    )

    print(max_memory)

    print(model_adapter.no_split_module_classes)
    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model_adapter.no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
