import pytest
import timm
import torch

from effsim.backbone import embed
from effsim.timing import count_gflops, env_info, sdpa_op, time_cpu, time_gpu


def tiny():
    return timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()


def test_count_gflops_linear():
    lin = torch.nn.Linear(100, 50)
    assert count_gflops(lin, torch.randn(8, 100)) == pytest.approx(2 * 100 * 50 / 1e9)


def test_time_cpu_restores_threads():
    m = tiny()
    before = torch.get_num_threads()
    t = time_cpu(lambda x: embed(m, x), torch.randn(1, 3, 224, 224), warmup=1, iters=4)
    assert t["median_ms"] > 0 and t["iqr_ms"] >= 0
    assert torch.get_num_threads() == before


def test_env_info_and_sdpa_op():
    m = tiny()
    info = env_info(m, "cpu")
    assert set(info) == {"device", "on_ac", "torch", "timm", "fused_attn"} and info["device"]
    assert "scaled_dot_product" in sdpa_op(lambda x: embed(m, x), torch.randn(1, 3, 224, 224))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_time_gpu():
    m = tiny().cuda()
    t = time_gpu(lambda x: embed(m, x), torch.randn(2, 3, 224, 224), "fp16", warmup=2, iters=5)
    assert t["median_ms"] > 0 and t["peak_mem_mb"] > 0
