"""GPU and single-core CPU latency, FLOPs, memory (SPEC §8.2)."""

import platform
import statistics
import time
from pathlib import Path

import timm
import torch
from torch.utils.flop_counter import FlopCounterMode


def _quartiles(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return statistics.median(xs), q[2] - q[0]


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def on_ac_power():
    supplies = list(Path("/sys/class/power_supply").glob("*/online"))
    return any(p.read_text().strip() == "1" for p in supplies) if supplies else None


@torch.inference_mode()
def sdpa_op(fn, x):
    """The attention kernel that actually runs, read from the profiler."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn(x)
    names = {e.key for e in prof.key_averages()}
    hits = sorted(n for n in names if "scaled_dot_product" in n and n != "aten::scaled_dot_product_attention")
    return ",".join(hits) or ("aten::scaled_dot_product_attention" if any("scaled_dot_product" in n for n in names)
                              else "explicit-matmul")


def env_info(model, device):
    return {
        "device": torch.cuda.get_device_name() if device == "cuda" else cpu_name(),
        "on_ac": on_ac_power(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "fused_attn": getattr(model.blocks[0].attn, "fused_attn", None),
    }


@torch.inference_mode()
def count_gflops(fn, x):
    with FlopCounterMode(display=False) as counter:
        fn(x)
    return counter.get_total_flops() / x.shape[0] / 1e9


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def time_gpu(fn, x, precision, warmup=25, iters=200):
    """Median and IQR in ms over CUDA-event timed iterations, plus peak memory in MB."""
    x = x.cuda()
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(x)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    median, iqr = _quartiles(times)
    return {"median_ms": median, "iqr_ms": iqr, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}


@torch.inference_mode()
def time_cpu(fn, x, warmup=20, iters=100, threads=1):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            times.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(previous)
    median, iqr = _quartiles(times)
    return {"median_ms": median, "iqr_ms": iqr}
