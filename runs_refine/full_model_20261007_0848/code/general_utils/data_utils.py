from __future__ import annotations
import logging
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_dataset(name, train=True) -> datasets.DatasetDict:
    import datasets

    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")
    
    properties = ds_properties[name]
    print(properties['path'])
    
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )
    
    print(ds)

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])
    
    if train:
        split = "train"
    else:
        if name == "c4":
            split = "validation"
        else:
            split = "test"
    
    logging.info("Loading dataset done")
    
    return ds[split]


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    import datasets

    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]

    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})
        

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader


def _squeeze_leading_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected a tensor in presampled calibration data, got: {type(tensor)}")

    tensor = tensor.detach().cpu()
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _infer_secondary_name(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu()

    if tensor.dtype == torch.bool:
        return "attention_mask"

    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return "attention_mask"

    if torch.all((flat == 0) | (flat == 1)):
        return "attention_mask"

    if torch.any(flat == -100):
        return "labels"

    return "attention_mask"


def _normalize_presampled_sample(sample, tuple_keys="auto") -> dict[str, torch.Tensor]:
    if isinstance(sample, dict):
        normalized = {k: _squeeze_leading_batch_dim(v) for k, v in sample.items() if torch.is_tensor(v)}
        if "input_ids" not in normalized:
            raise ValueError("Presampled dict sample must contain an 'input_ids' tensor.")
        return normalized

    if not isinstance(sample, (tuple, list)):
        raise TypeError(
            "Each presampled calibration sample must be either a dict or a tuple/list of tensors. "
            f"Got: {type(sample)}"
        )

    tensors = [_squeeze_leading_batch_dim(t) for t in sample]
    if len(tensors) == 0:
        raise ValueError("Encountered an empty tuple/list in presampled calibration data.")

    if tuple_keys == "auto":
        if len(tensors) == 1:
            key_names = ["input_ids"]
        elif len(tensors) == 2:
            key_names = ["input_ids", _infer_secondary_name(tensors[1])]
        elif len(tensors) == 3:
            second_name = _infer_secondary_name(tensors[1])
            third_name = "labels" if second_name != "labels" else "attention_mask"
            key_names = ["input_ids", second_name, third_name]
        else:
            raise ValueError(
                "Auto parsing only supports tuple/list samples of length 1, 2, or 3. "
                f"Got length {len(tensors)}. Please provide tuple_keys explicitly."
            )
    else:
        if isinstance(tuple_keys, str):
            key_names = [x.strip() for x in tuple_keys.split(",") if x.strip()]
        else:
            key_names = list(tuple_keys)

        if len(key_names) != len(tensors):
            raise ValueError(
                f"tuple_keys has {len(key_names)} entries, but each presampled sample has {len(tensors)} tensors."
            )

    normalized = {k: v for k, v in zip(key_names, tensors)}
    if "input_ids" not in normalized:
        raise ValueError("Presampled calibration sample is missing 'input_ids'.")
    return normalized


def _collate_presampled_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = samples[0].keys()
    batch = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values, dim=0)
    return batch


def prepare_presampled_dataloader(
    data_path: str,
    batch_size: int = 1,
    tuple_keys="auto",
) -> DataLoader[dict[str, torch.Tensor]]:
    logging.info(f"Loading presampled calibration data from: {data_path}")
    raw_data = torch.load(data_path, map_location="cpu")

    if not isinstance(raw_data, list):
        raise TypeError(
            "Expected presampled calibration data to be a Python list, "
            f"but got {type(raw_data)}."
        )

    if len(raw_data) == 0:
        raise ValueError("The presampled calibration data list is empty.")

    samples = [_normalize_presampled_sample(sample, tuple_keys=tuple_keys) for sample in raw_data]

    sample_keys = list(samples[0].keys())
    logging.info(
        "Loaded %d presampled calibration samples with keys: %s",
        len(samples),
        sample_keys,
    )

    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=_collate_presampled_batch)
    logging.info("Preparing presampled dataloader done")
    return loader
