"""Time encoders per SPEC §8.2 and write one fresh CSV per session.

    python scripts/time_models.py --device cpu                 # -> results/timing_cpu.csv
    python scripts/time_models.py --device cuda --res 224 448  # -> results/timing_gpu.csv (Colab)
"""

import argparse
import csv
import datetime
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from effsim.backbone import MODEL_NAME, embed  # noqa: E402
from effsim.timing import count_gflops, env_info, n_params, sdpa_op, time_cpu, time_gpu  # noqa: E402
from flop_model import encoder_gflops  # noqa: E402

FIELDS = ["date", "device", "on_ac", "torch", "timm", "fused_attn", "sdpa_op", "config", "res", "precision",
          "batch", "threads", "median_ms", "iqr_ms", "peak_mem_mb", "gflops", "flop_model_gflops",
          "flop_diff_pct", "params_m", "commit"]


def build(config, res):
    import timm

    if config != "none":
        raise NotImplementedError(f"config {config}: mechanisms arrive in P2")
    return timm.create_model(MODEL_NAME, pretrained=False, img_size=res).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--res", type=int, nargs="+", default=[224])
    ap.add_argument("--configs", nargs="+", default=["none"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or ROOT / "results" / f"timing_{a.device}.csv"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    order = list(a.configs)
    if a.device == "cuda":
        random.shuffle(order)
        if len(order) > 1:
            order = ["none"] + [c for c in order if c != "none"] + ["none"]
    runs = [(c, p, b) for c in order for p, b in
            ([("fp32", 1)] if a.device == "cpu" else [("fp16", 1), ("fp32", 1), ("fp16", 64), ("fp32", 64)])]

    rows = []
    for res in a.res:
        for config, precision, batch in runs:
            model = build(config, res).to(a.device)
            fn = lambda x: embed(model, x)  # noqa: E731
            x = torch.randn(batch, 3, res, res)
            info = env_info(model, a.device)
            gflops = count_gflops(fn, torch.randn(1, 3, res, res, device=a.device))
            expected = encoder_gflops(res, "mha", set())
            diff = 100 * (gflops / expected - 1)
            if a.device == "cpu":
                t = {**time_cpu(fn, x), "peak_mem_mb": None}
                threads = 1
            else:
                t = time_gpu(fn, x, precision)
                threads = None
            row = {"date": datetime.date.today().isoformat(), **info, "sdpa_op": sdpa_op(fn, x[:1].to(a.device)),
                   "config": config, "res": res, "precision": precision, "batch": batch, "threads": threads,
                   **t, "gflops": round(gflops, 3), "flop_model_gflops": round(expected, 3),
                   "flop_diff_pct": round(diff, 2), "params_m": round(n_params(model) / 1e6, 3), "commit": commit}
            rows.append(row)
            print(f"{config} {res}px {precision} b{batch}: {t['median_ms']:.1f} ms (IQR {t['iqr_ms']:.1f}), "
                  f"{gflops:.2f} GFLOPs vs model {expected:.2f} ({diff:+.2f}%)")
            if abs(diff) > 2:
                print(f"  FLOP check: counter and scripts/flop_model.py differ by {diff:+.2f}% (> 2%)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in FIELDS} for r in rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
