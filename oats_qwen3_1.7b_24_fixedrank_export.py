#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch

TARGET_PROJS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _sorted_oats_layer_checkpoints(checkpoint_dir: Path) -> list[Path]:
    ckpts = list(checkpoint_dir.glob("oats_chkpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No per-layer OATS checkpoints found under: {checkpoint_dir}. "
            "Expected files like oats_chkpt_0.pt, oats_chkpt_1.pt, ..."
        )

    def layer_idx(p: Path) -> int:
        stem = p.stem  # oats_chkpt_12
        return int(stem.rsplit("_", 1)[1])

    return sorted(ckpts, key=layer_idx)


def export_from_oats_layer_checkpoints(checkpoint_dir: Path, output_path: Path) -> None:
    """Export OATS Llama layer checkpoints to a SparseLoRA-style parameter dict.

    Mapping for an OATS-compressed nn.Linear:
        sp.weight   <- S   (sparse term)
        lr.l1.weight <- V  (rank, in_features)
        lr.l2.weight <- U  (out_features, rank)

    If a target module is still a plain nn.Linear in the saved state_dict, export the
    dense weight and make lr.l1/lr.l2 empty tensors for uniform downstream handling.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    output_path = output_path.resolve()

    layer_ckpts = _sorted_oats_layer_checkpoints(checkpoint_dir)

    param_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    exported_modules = 0
    missing_modules: list[str] = []

    for ckpt_path in layer_ckpts:
        layer_index = int(ckpt_path.stem.rsplit("_", 1)[1])
        state = torch.load(ckpt_path, map_location="cpu")

        for proj in TARGET_PROJS:
            export_prefix = f"model.layers.{layer_index}.{proj}"
            key_s = f"{proj}.S"
            key_v = f"{proj}.V"
            key_u = f"{proj}.U"
            key_w = f"{proj}.weight"

            if key_s in state and key_v in state and key_u in state:
                weight = state[key_s].detach().cpu().clone()
                lr1 = state[key_v].detach().cpu().clone()
                lr2 = state[key_u].detach().cpu().clone()
            elif key_w in state:
                weight = state[key_w].detach().cpu().clone()
                out_features, in_features = weight.shape
                lr1 = torch.empty((0, in_features), dtype=weight.dtype)
                lr2 = torch.empty((out_features, 0), dtype=weight.dtype)
            else:
                missing_modules.append(export_prefix)
                continue

            param_dict[f"{export_prefix}.sp.weight"] = weight
            param_dict[f"{export_prefix}.lr.l1.weight"] = lr1
            param_dict[f"{export_prefix}.lr.l2.weight"] = lr2
            exported_modules += 1

    if exported_modules == 0:
        raise RuntimeError(
            "Export produced zero modules. Make sure the checkpoint directory contains "
            "Llama OATS layer checkpoints."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(param_dict, output_path)

    print(f"Exported {exported_modules} projection modules to: {output_path}")
    if missing_modules:
        print("Warning: some expected projection modules were not found in the layer checkpoints:")
        for name in missing_modules:
            print(f"  - {name}")


def _build_pruning_args(rank_ratio: float, dtype: str, fixed_rank: int | None = None):
    from ml_collections import ConfigDict

    prune_hyper = {
                "unscaled": False,
                "rank_ratio": rank_ratio,
                "num_iters": 80,
                "compress": True,
                "prune_level": "row",
                "sparsity_type": "2:4",
            }
    if fixed_rank is not None:
        prune_hyper["fixed_rank"] = int(fixed_rank)

    args = ConfigDict(
        {
            "model": "qwen3-1.7b-base",
            "prune_type": "OATS",
            # NOTE:
            #   The supplement code asserts sparsity==0.5 whenever structured N:M sparsity is used.
            #   For OATS+N:M, the *actual* overall compression is controlled jointly by the N:M pattern
            #   and rank_ratio. So this field is mainly a switch to enter the structured branch.
            "sparsity": 0.5,
            "dtype": dtype,
            "distribute_model": True,
            "device": None,
            "cal_dataset": "c4",
            "cal_nsamples": 128,
            "cal_batch_size": 32,
            "cal_max_seqlen": 2048,
            "varied_seqlen": False,
            "use_owl": False,
            "seed": 0,
            "save_model": False,
            "eval_zero_shot": False,
            "eval_mmlu": False,
            "eval_ppl": False,
            "prune_hyper": prune_hyper,
            "presampled_data_path": "/home/zwang53/oats_code/calib_qwen3-1.7b.pth",
        }
    )
    return args


def prune_then_export(repo_root: Path, checkpoint_dir: Path, output_path: Path, rank_ratio: float, dtype: str, fixed_rank: int | None = None) -> None:
    repo_root = repo_root.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from run_prune import process_pruning_args, pruning_main

    args = _build_pruning_args(rank_ratio=rank_ratio, dtype=dtype, fixed_rank=fixed_rank)
    process_pruning_args(args)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_prefix = checkpoint_dir / "results"
    pruning_main(args, str(checkpoint_dir), str(results_prefix))
    export_from_oats_layer_checkpoints(checkpoint_dir=checkpoint_dir, output_path=output_path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Prune Llama-3 8B with OATS 2:4 and export a SparseLoRA-style param dict. Supports either rank_ratio or a fixed low-rank dimension per layer.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the extracted oats_code directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory used by OATS to save prune_chkpt.pt and oats_chkpt_*.pt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the exported param_dict .pt file.",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=0.10,
        help="OATS rank ratio. For 2:4 sparse terms, smaller values keep overall compression closer to 50%%.",
    )
    parser.add_argument(
        "--fixed-rank",
        type=int,
        default=None,
        help="If set, request the same low-rank dimension for every pruned projection (requires applying the fixed-rank patch to the OATS repo).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Computation dtype used during pruning.",
    )
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="Do not run pruning; only export from an existing checkpoint directory.",
    )
    args = parser.parse_args()

    if args.skip_prune:
        export_from_oats_layer_checkpoints(args.checkpoint_dir, args.output)
    else:
        prune_then_export(
            repo_root=args.repo_root,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            rank_ratio=args.rank_ratio,
            dtype=args.dtype,
            fixed_rank=args.fixed_rank,
        )


if __name__ == "__main__":
    main()
