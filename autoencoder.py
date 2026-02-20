"""Autoencoder for anomaly detection via reconstruction error.

Matches the U-Net checkpoint structure from model.pth:
  enc1-4 → bottleneck → dec4-1 with skip connections.
  Input: 128×128, latent_dim: 512.

Backends:
  - "torch" : PyTorch (Mac/dev — requires ARMv8.2+ on aarch64)
  - "onnx"  : ONNX Runtime (RPi 4B safe — supports Cortex-A72 / ARMv8.0)

IMPORTANT: torch is NEVER imported at module level.  On Cortex-A72 (RPi 4B),
`import torch` triggers SIGILL (illegal instruction) because pip's aarch64
wheels use ARMv8.2+ ops (SDOT/UDOT).  All torch usage is deferred to
_AnomalyDetectorTorch.__init__() so that choosing ae_backend="onnx" avoids
touching torch entirely.
"""

import numpy as np
import cv2
from config import Config

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


# ═══════════════════════════════════════════════════════════════
# PyTorch model builder (lazy — called only from Torch backend)
# ═══════════════════════════════════════════════════════════════

def _build_torch_model():
    """Build the UNetAutoencoder. Imports torch HERE, not at module level."""
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            )
        def forward(self, x):
            return torch.relu(x + self.block(x))

    def _enc_block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(True), ResBlock(c_out),
        )

    def _dec_block(c_in, c_out):
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(True),
        )

    def _fuse_block(c_skip, c_out):
        return nn.Sequential(
            nn.Conv2d(c_skip + c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(True), ResBlock(c_out),
        )

    class UNetAutoencoder(nn.Module):
        def __init__(self, img_size=128, latent_dim=512, in_channels=3):
            super().__init__()
            self.spatial = img_size // 16
            self.enc1 = _enc_block(in_channels, 32)
            self.enc2 = _enc_block(32, 64)
            self.enc3 = _enc_block(64, 128)
            self.enc4 = _enc_block(128, 256)
            flat = 256 * self.spatial * self.spatial
            self.bottleneck_encode = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim))
            self.bottleneck_decode = nn.Sequential(nn.Linear(latent_dim, flat), nn.ReLU(True))
            self.dec4 = _dec_block(256, 128)
            self.dec4_fuse = _fuse_block(128, 128)
            self.dec3 = _dec_block(128, 64)
            self.dec3_fuse = _fuse_block(64, 64)
            self.dec2 = _dec_block(64, 32)
            self.dec2_fuse = _fuse_block(32, 32)
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(True),
                nn.Conv2d(16, in_channels, 3, padding=1),
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            z = self.bottleneck_encode(e4)
            d = self.bottleneck_decode(z).view_as(e4)
            d = self.dec4(d)
            d = self.dec4_fuse(torch.cat([e3, d], dim=1))
            d = self.dec3(d)
            d = self.dec3_fuse(torch.cat([e2, d], dim=1))
            d = self.dec2(d)
            d = self.dec2_fuse(torch.cat([e1, d], dim=1))
            d = self.dec1(d)
            return torch.sigmoid(d)

    return UNetAutoencoder


# ═══════════════════════════════════════════════════════════════
# ONNX Runtime backend (RPi 4B / Cortex-A72 safe)
# ═══════════════════════════════════════════════════════════════

class _AnomalyDetectorONNX:
    """ONNX Runtime inference — no PyTorch dependency at runtime."""

    AE_SIZE = 128

    def __init__(self, cfg: Config):
        if not _HAS_ORT:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        onnx_path = cfg.autoencoder_onnx_path
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[AE-ONNX] Loaded {onnx_path}")

    def infer(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.AE_SIZE, self.AE_SIZE))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        error = np.abs(blob - output).mean(axis=1).squeeze(0)
        mean_error = float(error.mean())
        error_map = cv2.resize(error, (w, h))
        return error_map, mean_error


# ═══════════════════════════════════════════════════════════════
# PyTorch backend (Mac / dev)
# ═══════════════════════════════════════════════════════════════

class _AnomalyDetectorTorch:
    """PyTorch inference — uses MPS on Apple Silicon, CPU otherwise.
    
    All torch imports happen HERE, not at module level, so that
    ae_backend='onnx' never touches torch (avoids SIGILL on Cortex-A72).
    """

    AE_SIZE = 128

    def __init__(self, cfg: Config):
        import torch as _torch
        self._torch = _torch

        if _torch.backends.mps.is_available():
            self.device = _torch.device("mps")
        else:
            self.device = _torch.device("cpu")

        UNetAutoencoder = _build_torch_model()
        self.model = UNetAutoencoder(img_size=128, latent_dim=512, in_channels=3).to(self.device)
        try:
            ckpt = _torch.load(cfg.autoencoder_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            epoch = ckpt.get("epoch", "?")
            val_loss = ckpt.get("val_loss", 0)
            print(f"[AE] Loaded {cfg.autoencoder_path} (epoch {epoch}, val_loss {val_loss:.4f})")
        except FileNotFoundError:
            print(f"[AE] WARNING: {cfg.autoencoder_path} not found, using random weights")
        self.model.eval()

    def infer(self, frame: np.ndarray) -> tuple:
        _torch = self._torch
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.AE_SIZE, self.AE_SIZE))
        tensor = _torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with _torch.inference_mode():
            output = self.model(tensor)
        error = _torch.abs(tensor - output).mean(dim=1).squeeze(0).cpu().numpy()
        mean_error = float(error.mean())
        error_map = cv2.resize(error, (w, h))
        return error_map, mean_error


# ═══════════════════════════════════════════════════════════════
# Public factory
# ═══════════════════════════════════════════════════════════════

def AnomalyDetector(cfg: Config):
    """
    Factory — returns the correct AE backend:
      cfg.ae_backend == "onnx"  → ONNX Runtime  (RPi safe)
      cfg.ae_backend == "torch" → PyTorch        (Mac/dev)
    """
    backend = getattr(cfg, "ae_backend", "torch")
    if backend == "onnx":
        return _AnomalyDetectorONNX(cfg)
    else:
        return _AnomalyDetectorTorch(cfg)
