import torch

from worldsync.lingbot_camera_audit import audit_rays


def test_fp32_close_rays_may_straddle_bf16_rounding_boundary():
    midpoint = 0.501953125
    native = torch.tensor([midpoint - 1e-7], dtype=torch.float32)
    online = torch.tensor([midpoint + 1e-7], dtype=torch.float32)
    report = audit_rays(online, native, native.bfloat16())
    assert report["within_tolerance"]
    assert not report["tensor_identical"]
    assert report["native_cast_identical"]
    assert report["fp32_max_absolute_error"] < 1e-5
    assert report["max_absolute_error"] > 1e-3


def test_audit_rejects_real_input_drift_and_changed_native_tensor():
    native = torch.tensor([.5], dtype=torch.float32)
    assert not audit_rays(native + .002, native, native.bfloat16())["within_tolerance"]
    assert not audit_rays(native, native, (native + .002).bfloat16())["within_tolerance"]
