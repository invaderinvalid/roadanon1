#!/usr/bin/env python3
"""
Quantize the ONNX autoencoder model to INT8 (dynamic quantization).

Reduces model from ~76 MB → ~19 MB and speeds inference 2-4× on ARM NEON.
No calibration dataset needed — uses dynamic quantization.

Usage:
    python tools/quantize_onnx.py
    python tools/quantize_onnx.py --input models/model.onnx --output models/model_int8.onnx
    python tools/quantize_onnx.py --verify   # compare accuracy vs float32
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def quantize(input_path: str, output_path: str):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.preprocess import quant_pre_process

    print(f"[Quantize] Input:  {input_path} "
          f"({os.path.getsize(input_path) / 1024:.0f} KB)")

    # Check for external data file
    data_path = input_path + ".data"
    has_external = os.path.exists(data_path)
    if has_external:
        total = os.path.getsize(input_path) + os.path.getsize(data_path)
        print(f"[Quantize] External weights: {data_path} "
              f"({os.path.getsize(data_path) / 1024 / 1024:.1f} MB)")
        print(f"[Quantize] Total FP32 size: {total / 1024 / 1024:.1f} MB")

    # Step 1: Internalize external data so quant tools can handle it
    print("[Quantize] Loading & internalizing model...")
    model = onnx.load(input_path, load_external_data=True)
    tmp_path = output_path + ".tmp.onnx"
    onnx.save(model, tmp_path)
    del model

    # Step 2: ORT recommended preprocessing (shape inference, optimization, etc.)
    preproc_path = output_path + ".preproc.onnx"
    print("[Quantize] Running ORT preprocessing...")
    try:
        quant_pre_process(tmp_path, preproc_path)
        os.remove(tmp_path)
        src_path = preproc_path
    except Exception as e:
        print(f"[Quantize] Preprocessing failed ({e}), using raw model")
        src_path = tmp_path

    print(f"[Quantize] Preprocessed → {os.path.getsize(src_path) / 1024 / 1024:.1f} MB")

    print("[Quantize] Quantizing FP32 → INT8 (dynamic)...")
    t0 = time.time()

    quantize_dynamic(
        model_input=src_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    # Cleanup temp files
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(preproc_path):
        os.remove(preproc_path)

    elapsed = time.time() - t0
    out_size = os.path.getsize(output_path)
    # Check if quantized model also has external data
    out_data = output_path + ".data"
    if os.path.exists(out_data):
        out_size += os.path.getsize(out_data)

    print(f"[Quantize] ✓ Done in {elapsed:.1f}s")
    print(f"[Quantize] Output: {output_path} ({out_size / 1024 / 1024:.1f} MB)")

    if has_external:
        ratio = total / max(out_size, 1)
        print(f"[Quantize] Compression: {ratio:.1f}×")


def verify(fp32_path: str, int8_path: str, n_samples: int = 10):
    import numpy as np
    import onnxruntime as ort

    print(f"\n[Verify] Comparing FP32 vs INT8 ({n_samples} random inputs)...")

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2

    sess_fp32 = ort.InferenceSession(fp32_path, opts,
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, opts,
                                     providers=["CPUExecutionProvider"])

    input_name = sess_fp32.get_inputs()[0].name
    shape = sess_fp32.get_inputs()[0].shape  # [1, 3, 128, 128]

    max_diffs = []
    mean_diffs = []
    fp32_times = []
    int8_times = []

    for i in range(n_samples):
        inp = np.random.rand(*shape).astype(np.float32)

        t0 = time.time()
        out_fp32 = sess_fp32.run(None, {input_name: inp})[0]
        fp32_times.append(time.time() - t0)

        t0 = time.time()
        out_int8 = sess_int8.run(None, {input_name: inp})[0]
        int8_times.append(time.time() - t0)

        diff = np.abs(out_fp32 - out_int8)
        max_diffs.append(float(diff.max()))
        mean_diffs.append(float(diff.mean()))

    avg_fp32 = sum(fp32_times) / len(fp32_times) * 1000
    avg_int8 = sum(int8_times) / len(int8_times) * 1000

    print(f"[Verify] Max diff:  {max(max_diffs):.6f} (avg {sum(max_diffs)/len(max_diffs):.6f})")
    print(f"[Verify] Mean diff: {max(mean_diffs):.6f} (avg {sum(mean_diffs)/len(mean_diffs):.6f})")
    print(f"[Verify] FP32 avg:  {avg_fp32:.1f} ms/inference")
    print(f"[Verify] INT8 avg:  {avg_int8:.1f} ms/inference")
    print(f"[Verify] Speedup:   {avg_fp32 / max(avg_int8, 0.01):.2f}×")

    if max(max_diffs) < 0.05:
        print("[Verify] ✓ INT8 accuracy acceptable for anomaly detection")
    else:
        print("[Verify] ⚠ Large diff — test on real data before deploying")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", default="models/model.onnx",
                        help="FP32 ONNX model path")
    parser.add_argument("--output", default="models/model_int8.onnx",
                        help="INT8 output path")
    parser.add_argument("--verify", action="store_true",
                        help="Compare accuracy & speed vs FP32")
    args = parser.parse_args()

    quantize(args.input, args.output)

    if args.verify:
        verify(args.input, args.output)


if __name__ == "__main__":
    main()
