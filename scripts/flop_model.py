import argparse

D, H, MLP_RATIO, PATCH = 768, 12, 4, 16


def block_macs(n, mixer, r=2, frac=0.5):
    mlp = 2 * n * D * MLP_RATIO * D
    if mixer == "mha":
        mix = 4 * n * D * D + 2 * n * n * D
    elif mixer == "meta":
        mix = 0
    elif mixer == "pyra":
        g = round((n - 1) ** 0.5)
        nk = (g // r) ** 2 + 1
        mix = n * D * D + (n - 1) * D * D + 2 * nk * D * D + 2 * n * nk * D + n * D * D
    elif mixer == "moh_ideal":
        mix = frac * n * D * D + 2 * n * D * D + frac * 2 * n * n * D + frac * n * D * D + n * D * H
    elif mixer == "moh_dense":
        mix = 4 * n * D * D + 2 * n * n * D + n * D * H
    else:
        raise ValueError(mixer)
    return mix + mlp


def encoder_gflops(res, mixer, layers, **kw):
    g = res // PATCH
    n = g * g + 1
    patch = g * g * 3 * PATCH * PATCH * D
    total = patch + sum(block_macs(n, mixer if i in layers else "mha", **kw) for i in range(12))
    return 2 * total / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, nargs="+", default=[224, 448, 672])
    a = p.parse_args()
    configs = {"last-4": set(range(8, 12)), "first-4": set(range(4)), "all-12": set(range(12))}
    print("| res | tokens | quad-attn share | baseline GFLOPs | " + " | ".join(
        f"{m} {c}" for m in ("meta", "pyra", "moh_ideal", "moh_dense") for c in configs) + " |")
    for res in a.res:
        g = res // PATCH
        n = g * g + 1
        share = 2 * n * n * D / block_macs(n, "mha")
        base = encoder_gflops(res, "mha", set())
        cells = []
        for m in ("meta", "pyra", "moh_ideal", "moh_dense"):
            for layers in configs.values():
                cells.append(f"{encoder_gflops(res, m, layers) / base - 1:+.1%}")
        print(f"| {res} | {n} | {share:.1%} | {base:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
