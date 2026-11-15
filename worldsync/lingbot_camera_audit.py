"""Compare camera arithmetic before BF16 rounding, retaining native inputs."""
import torch


def audit_rays(online_fp32, native_fp32, native_bf16):
    expected_bf16 = native_fp32.to(native_bf16.dtype)
    exact_native_cast = torch.equal(expected_bf16, native_bf16)
    return {
        "audit_version": "fp32-and-native-cast-v2",
        "max_absolute_error": float((online_fp32.to(native_bf16.dtype).float() - native_bf16.float()).abs().max()),
        "fp32_max_absolute_error": float((online_fp32 - native_fp32).abs().max()),
        "tensor_identical": torch.equal(online_fp32.to(native_bf16.dtype), native_bf16),
        "native_cast_identical": exact_native_cast,
        "within_tolerance": exact_native_cast and torch.allclose(online_fp32, native_fp32, atol=1e-5, rtol=1e-5),
    }
