"""Convert autoencoder to FP16 ONNX for faster RPi inference.

No tensorflow needed — uses onnxruntime + onnx only.

Usage:
    python convert_ae_tflite.py

Output:
    models/autoencoder_fp16.onnx
"""

import os
import sys
import numpy as np


def export_onnx_from_pth(pth_path, onnx_path):
    """Export PyTorch model.pth → ONNX."""
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from autoencoder import _build_torch_model

    UNetAutoencoder = _build_torch_model()
    model = UNetAutoencoder(img_size=128, latent_dim=512, in_channels=3)

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[1/2] Exported ONNX: {onnx_path} ({size_kb:.0f} KB)")


def convert_fp16(onnx_path, fp16_path):
    """Convert ONNX FP32 → FP16 using onnxconverter-common."""
    import onnx
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("Installing onnxconverter-common...")
        os.system(f"{sys.executable} -m pip install onnxconverter-common")
        from onnxconverter_common import float16

    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fp16_path)

    size_kb = os.path.getsize(fp16_path) / 1024
    print(f"[2/2] FP16 ONNX: {fp16_path} ({size_kb:.0f} KB)")


def main():
    pth_path = "models/model.pth"
    onnx_fp32 = "models/model.onnx"
    onnx_fp16 = "models/autoencoder_fp16.onnx"

    print("=" * 50)
    print("  Autoencoder → FP16 ONNX")
    print("=" * 50)

    # Step 1: Get ONNX model (export from .pth or use existing)
    if os.path.exists(onnx_fp32):
        print(f"[1/2] Using existing ONNX: {onnx_fp32}")
    elif os.path.exists(pth_path):
        export_onnx_from_pth(pth_path, onnx_fp32)
    else:
        print(f"ERROR: neither {pth_path} nor {onnx_fp32} found")
        sys.exit(1)

    # Step 2: Convert to FP16
    convert_fp16(onnx_fp32, onnx_fp16)

    orig_size = os.path.getsize(onnx_fp32) / 1024
    fp16_size = os.path.getsize(onnx_fp16) / 1024
    print(f"\n✓ Done!")
    print(f"  Original: {orig_size:.0f} KB")
    print(f"  FP16:     {fp16_size:.0f} KB ({fp16_size/orig_size*100:.0f}%)")


if __name__ == "__main__":
    main()
