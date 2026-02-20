#!/usr/bin/env python3
"""
Export the U-Net autoencoder from model.pth → model.onnx

Usage:
    python tools/export_onnx.py                        # defaults
    python tools/export_onnx.py --input models/model.pth --output models/model.onnx
    python tools/export_onnx.py --simplify              # pip install onnx-simplifier
"""

import os
import sys
import argparse

# Allow import from parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def export(input_path: str, output_path: str, simplify: bool = False):
    from autoencoder import UNetAutoencoder

    print(f"[Export] Loading {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    model = UNetAutoencoder(img_size=128, latent_dim=512, in_channels=3)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128)

    print(f"[Export] Exporting to ONNX → {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed batch=1
    )
    print(f"[Export] ✓ Saved {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("[Export] Simplifying...")
            model_onnx = onnx.load(output_path)
            model_simp, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(model_simp, output_path)
                print(f"[Export] ✓ Simplified ({os.path.getsize(output_path) / 1024:.0f} KB)")
            else:
                print("[Export] Simplification failed, keeping original")
        except ImportError:
            print("[Export] onnx-simplifier not installed, skipping (pip install onnx-simplifier)")

    # Verify
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path)
        inp = np.random.randn(1, 3, 128, 128).astype(np.float32)
        out = sess.run(None, {"input": inp})[0]
        print(f"[Verify] ONNX Runtime OK — input {inp.shape} → output {out.shape}")

        # Compare with PyTorch
        with torch.no_grad():
            pt_out = model(torch.from_numpy(inp)).numpy()
        diff = float(np.abs(pt_out - out).max())
        print(f"[Verify] Max diff vs PyTorch: {diff:.6f}")
        if diff < 1e-4:
            print("[Verify] ✓ Match within tolerance")
        else:
            print(f"[Verify] ⚠ Difference is {diff:.6f} (may still be acceptable)")
    except ImportError:
        print("[Verify] Install onnxruntime to verify: pip install onnxruntime")


def main():
    parser = argparse.ArgumentParser(description="Export U-Net autoencoder to ONNX")
    parser.add_argument("--input", "-i", default="models/model.pth",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", "-o", default="models/model.onnx",
                        help="Output ONNX path")
    parser.add_argument("--simplify", action="store_true",
                        help="Apply onnx-simplifier (optional)")
    args = parser.parse_args()
    export(args.input, args.output, args.simplify)


if __name__ == "__main__":
    main()
