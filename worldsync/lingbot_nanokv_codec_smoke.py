"""Check released codec arithmetic/kernels before loading a video backbone."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from worldsync.lingbot_nanokv import NativeNanoKV, validate_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    adapter = NativeNanoKV(SimpleNamespace(model=torch.nn.Module()),
        source_root=args.source_root, codec_path=args.codec, capacity_frames=58)
    summary = {'scope': 'Synthetic codec plumbing only, not video quality or training',
               'device': str(device), 'torch': str(torch.__version__), 'branches': {}, 'status': 'running'}
    try:
        adapter.packed = validate_codec(args.codec)
        adapter._helpers()
        generator = torch.Generator(device=device).manual_seed(127)
        for target in ['k', 'v']:
            value = torch.randn((1, 128, 40, 128), device=device,
                                dtype=torch.bfloat16, generator=generator)
            payload = adapter.codec.encode_tensor(value, target=target, layer=0, sp_rank=0,
                basis_root='verified-packed-codec', rank=640, clusters=64, iters=2)
            restored = adapter.codec.decode_tensor(payload, dtype=value.dtype, device=device)
            if restored.shape != value.shape or restored.dtype != value.dtype or not torch.isfinite(restored).all():
                raise ValueError('Codec output shape, dtype or finiteness failed')
            summary['branches'][target] = {'shape': list(restored.shape), 'finite': True,
                'relative_l2_synthetic_only': float((restored.float()-value.float()).norm()/value.float().norm())}
        summary['status'] = 'completed'
    finally:
        adapter.close()
        summary['adapter_closed'] = adapter.metadata()['closed']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
