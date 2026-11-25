#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def make_24_mask(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    rows, cols = shape
    if cols % 4 != 0:
        raise ValueError(f"2:4 mask requires cols % 4 == 0, got {shape}")
    base = torch.tensor([1, 1, 0, 0], device=device, dtype=torch.bool)
    return base.repeat(rows, cols // 4)


def apply_24_mask(t: torch.Tensor) -> torch.Tensor:
    mask = make_24_mask(tuple(t.shape), t.device)
    return t.masked_fill(~mask, 0)


def maybe_to_semi_structured(t: torch.Tensor):
    from torch.sparse import to_sparse_semi_structured
    return to_sparse_semi_structured(t)


def supports_semi_structured(device: torch.device, dtype: torch.dtype, shape: Tuple[int, int]) -> Tuple[bool, str]:
    if device.type != "cuda":
        return False, "需要 CUDA"
    if dtype not in (torch.float16, torch.bfloat16):
        return False, f"dtype={dtype} 不支持真实 2:4 加速，建议 fp16/bf16"
    rows, cols = shape
    if rows % 64 != 0 or cols % 64 != 0:
        return False, f"shape={shape} 不是 64 的倍数，PyTorch 半结构化 sparse 路径不支持"
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        return False, f"SM capability={major}.{minor}，需要 8.0+"
    return True, "ok"


@dataclass
class BenchRow:
    case: str
    rank: int
    effective_rank: int
    compute_rank: int
    ms_median: float
    ms_mean: float
    tflops: float
    note: str


class CUDATimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def timeit(self, fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> List[float]:
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            self.start.record()
            _ = fn()
            self.end.record()
            torch.cuda.synchronize()
            times.append(self.start.elapsed_time(self.end))
        return times


def cpu_timeit(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        _ = fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


def benchmark_case(fn: Callable[[], torch.Tensor], warmup: int, iters: int, device: torch.device) -> Tuple[float, float]:
    if device.type == "cuda":
        timer = CUDATimer()
        times = timer.timeit(fn, warmup, iters)
    else:
        times = cpu_timeit(fn, warmup, iters)
    return statistics.median(times), statistics.mean(times)


def dense_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def lowrank_flops(m: int, k: int, n: int, r: int) -> float:
    return 2.0 * (m * k * r + m * r * n)


def tflops_from_ms(flops: float, ms: float) -> float:
    sec = ms / 1e3
    return flops / sec / 1e12


def build_inputs(batch: int, hidden: int, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch, hidden, device=device, dtype=dtype)
    # F.linear(input, weight) expects weight shape [out_features, in_features].
    w_dense = torch.randn(hidden, hidden, device=device, dtype=dtype)
    w_24_dense = apply_24_mask(w_dense.clone())
    return {"x": x, "w_dense": w_dense, "w_24_dense": w_24_dense}


def maybe_compile(fn: Callable[[], torch.Tensor], enabled: bool, fullgraph: bool) -> Callable[[], torch.Tensor]:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(fn, fullgraph=fullgraph)
    return fn


def try_benchmark(
        case: str,
        rank: int,
        effective_rank: int,
        compute_rank: int,
        fn: Callable[[], torch.Tensor],
        flops: float,
        warmup: int,
        iters: int,
        device: torch.device,
        note: str,
) -> BenchRow:
    try:
        ms_med, ms_mean = benchmark_case(fn, warmup, iters, device)
        return BenchRow(
            case=case,
            rank=rank,
            effective_rank=effective_rank,
            compute_rank=compute_rank,
            ms_median=ms_med,
            ms_mean=ms_mean,
            tflops=tflops_from_ms(flops, ms_med),
            note=note,
        )
    except Exception as e:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return BenchRow(
            case=f"{case}_UNSUPPORTED",
            rank=rank,
            effective_rank=effective_rank,
            compute_rank=compute_rank,
            ms_median=float("nan"),
            ms_mean=float("nan"),
            tflops=float("nan"),
            note=f"{note}; runtime error: {type(e).__name__}: {e}",
        )


def run_bench(
        batch: int,
        hidden: int,
        ranks: List[int],
        dtype: torch.dtype,
        device: torch.device,
        warmup: int,
        iters: int,
        use_compile_dense: bool,
        use_compile_sparse: bool,
        out_csv: Optional[str],
) -> List[BenchRow]:
    inputs = build_inputs(batch, hidden, dtype, device)
    x = inputs["x"]
    w_dense = inputs["w_dense"]
    w_24_dense = inputs["w_24_dense"]

    rows: List[BenchRow] = []

    # Dense baseline via F.linear, weight=[out,in].
    fn_dense = maybe_compile(lambda: F.linear(x, w_dense, bias=None), enabled=use_compile_dense, fullgraph=True)
    rows.append(try_benchmark(
        case="dense_full_linear",
        rank=0,
        effective_rank=hidden,
        compute_rank=hidden,
        fn=fn_dense,
        flops=dense_flops(batch, hidden, hidden),
        warmup=warmup,
        iters=iters,
        device=device,
        note="baseline dense F.linear([512,4096], [4096,4096])" + ("; compiled" if use_compile_dense else "; eager"),
    ))

    # Full 2:4 sparse via F.linear.
    ok, reason = supports_semi_structured(device, dtype, tuple(w_24_dense.shape))
    if ok:
        w_24_sparse = maybe_to_semi_structured(w_24_dense)
        sparse_note = "real semi-structured sparse kernel via F.linear"
        fn_sparse_base = lambda: F.linear(x, w_24_sparse, bias=None)
        fn_sparse = maybe_compile(fn_sparse_base, enabled=use_compile_sparse, fullgraph=False)
        if not use_compile_sparse:
            sparse_note += "; eager only to avoid torch.compile sparse dispatch failures"
        else:
            sparse_note += "; compile requested"
        rows.append(try_benchmark(
            case="sparse24_full_linear",
            rank=0,
            effective_rank=hidden,
            compute_rank=hidden,
            fn=fn_sparse,
            flops=dense_flops(batch, hidden, hidden),
            warmup=warmup,
            iters=iters,
            device=device,
            note=sparse_note,
        ))
    else:
        rows.append(BenchRow(
            case="sparse24_full_linear_UNSUPPORTED",
            rank=0,
            effective_rank=hidden,
            compute_rank=hidden,
            ms_median=float("nan"),
            ms_mean=float("nan"),
            tflops=float("nan"),
            note=reason,
        ))

    for r in ranks:
        # Low-rank expressed as two linears:
        #   y = F.linear(F.linear(x, w1), w2)
        # where w1.shape=[r, hidden], w2.shape=[hidden, r].
        w1 = torch.randn(r, hidden, device=device, dtype=dtype)
        w2 = torch.randn(hidden, r, device=device, dtype=dtype)

        fn_lr = maybe_compile(
            lambda w1=w1, w2=w2: F.linear(F.linear(x, w1, bias=None), w2, bias=None),
            enabled=use_compile_dense,
            fullgraph=True,
        )
        rows.append(try_benchmark(
            case="lowrank_dense_linear",
            rank=r,
            effective_rank=r,
            compute_rank=r,
            fn=fn_lr,
            flops=lowrank_flops(batch, hidden, hidden, r),
            warmup=warmup,
            iters=iters,
            device=device,
            note="F.linear([512,4096],[r,4096]) -> F.linear([512,r],[4096,r])" + ("; compiled" if use_compile_dense else "; eager"),
        ))

        r_compute = r if r % 64 == 0 else round_up(r, 64)
        w1_pad = torch.zeros(r_compute, hidden, device=device, dtype=dtype)
        w2_pad = torch.zeros(hidden, r_compute, device=device, dtype=dtype)
        w1_pad[:r, :] = w1
        w2_pad[:, :r] = w2

        w1_24 = apply_24_mask(w1_pad)
        w2_24 = apply_24_mask(w2_pad)

        note_parts = []
        if r_compute != r:
            note_parts.append(f"rank 从 {r} pad 到 {r_compute} 才能走真实 2:4 sparse")

        ok_w1, reason_w1 = supports_semi_structured(device, dtype, tuple(w1_24.shape))
        ok_w2, reason_w2 = supports_semi_structured(device, dtype, tuple(w2_24.shape))
        if ok_w1 and ok_w2:
            w1_sparse = maybe_to_semi_structured(w1_24)
            w2_sparse = maybe_to_semi_structured(w2_24)
            fn_lr_sparse_base = lambda w1_sparse=w1_sparse, w2_sparse=w2_sparse: F.linear(
                F.linear(x, w1_sparse, bias=None), w2_sparse, bias=None
            )
            fn_lr_sparse = maybe_compile(fn_lr_sparse_base, enabled=use_compile_sparse, fullgraph=False)

            if not use_compile_sparse:
                note_parts.append("sparse case 默认 eager，避免 torch.compile + SparseSemiStructuredTensor 失败")
            else:
                note_parts.append("compile requested")

            rows.append(try_benchmark(
                case="lowrank_sparse24x2_linear",
                rank=r,
                effective_rank=r,
                compute_rank=r_compute,
                fn=fn_lr_sparse,
                flops=lowrank_flops(batch, hidden, hidden, r_compute),
                warmup=warmup,
                iters=iters,
                device=device,
                note="; ".join(note_parts) if note_parts else "two sparse F.linear ops",
            ))
        else:
            note = "; ".join(note_parts + [f"W1: {reason_w1}", f"W2: {reason_w2}"])
            rows.append(BenchRow(
                case="lowrank_sparse24x2_linear_UNSUPPORTED",
                rank=r,
                effective_rank=r,
                compute_rank=r_compute,
                ms_median=float("nan"),
                ms_mean=float("nan"),
                tflops=float("nan"),
                note=note,
            ))

    if out_csv and rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    return rows


def format_table(rows: List[BenchRow]) -> str:
    headers = ["case", "rank", "eff_r", "comp_r", "median_ms", "mean_ms", "TFLOP/s", "note"]
    table = [headers]
    for r in rows:
        table.append([
            r.case,
            str(r.rank),
            str(r.effective_rank),
            str(r.compute_rank),
            f"{r.ms_median:.4f}" if math.isfinite(r.ms_median) else "NA",
            f"{r.ms_mean:.4f}" if math.isfinite(r.ms_mean) else "NA",
            f"{r.tflops:.3f}" if math.isfinite(r.tflops) else "NA",
            r.note,
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    out = []
    for ridx, row in enumerate(table):
        out.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ridx == 0:
            out.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2:4 sparse vs low-rank dense vs sparse low-rank using F.linear")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="Compile dense linear cases. Sparse linear cases stay eager unless --compile-sparse is also set.")
    parser.add_argument("--compile-sparse", action="store_true", help="Also try torch.compile on semi-structured sparse linear cases.")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    rows = run_bench(
        batch=args.batch,
        hidden=args.hidden,
        ranks=args.ranks,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        use_compile_dense=args.compile,
        use_compile_sparse=args.compile_sparse,
        out_csv=args.csv or None,
    )

    print(f"torch={torch.__version__}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"capability={torch.cuda.get_device_capability(device)}")
    print(f"dtype={dtype}")
    print(f"compile_dense={args.compile}")
    print(f"compile_sparse={args.compile_sparse}")
    print()
    print(format_table(rows))


if __name__ == "__main__":
    main()
