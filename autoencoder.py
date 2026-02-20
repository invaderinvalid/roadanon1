"""Autoencoder for anomaly detection via reconstruction error.

Matches the U-Net checkpoint structure from model.pth:
  enc1-4 → bottleneck → dec4-1 with skip connections.
  Input: 128×128, latent_dim: 512.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from config import Config


class ResBlock(nn.Module):
    """Two-conv residual block with BatchNorm."""

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
        nn.BatchNorm2d(c_out), nn.ReLU(True),
        ResBlock(c_out),
    )


def _dec_block(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(True),
    )


def _fuse_block(c_skip, c_out):
    return nn.Sequential(
        nn.Conv2d(c_skip + c_out, c_out, 1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(True),
        ResBlock(c_out),
    )


class UNetAutoencoder(nn.Module):
    """U-Net style autoencoder matching model.pth checkpoint."""

    def __init__(self, img_size=128, latent_dim=512, in_channels=3):
        super().__init__()
        self.spatial = img_size // 16  # 8 for 128

        # Encoder
        self.enc1 = _enc_block(in_channels, 32)   # → 32×64×64
        self.enc2 = _enc_block(32, 64)             # → 64×32×32
        self.enc3 = _enc_block(64, 128)            # → 128×16×16
        self.enc4 = _enc_block(128, 256)           # → 256×8×8

        flat = 256 * self.spatial * self.spatial
        self.bottleneck_encode = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim))
        self.bottleneck_decode = nn.Sequential(nn.Linear(latent_dim, flat), nn.ReLU(True))

        # Decoder (with skip connections)
        self.dec4 = _dec_block(256, 128)     # → 128×16×16
        self.dec4_fuse = _fuse_block(128, 128)  # cat(enc3, dec4) → 128

        self.dec3 = _dec_block(128, 64)      # → 64×32×32
        self.dec3_fuse = _fuse_block(64, 64)

        self.dec2 = _dec_block(64, 32)       # → 32×64×64
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


class AnomalyDetector:
    """Wraps autoencoder for frame-level anomaly scoring."""

    AE_SIZE = 128  # model input size (from checkpoint config)

    def __init__(self, cfg: Config):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = UNetAutoencoder(img_size=128, latent_dim=512, in_channels=3).to(self.device)
        try:
            ckpt = torch.load(cfg.autoencoder_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            epoch = ckpt.get("epoch", "?")
            val_loss = ckpt.get("val_loss", 0)
            print(f"[AE] Loaded {cfg.autoencoder_path} (epoch {epoch}, val_loss {val_loss:.4f})")
        except FileNotFoundError:
            print(f"[AE] WARNING: {cfg.autoencoder_path} not found, using random weights")
        self.model.eval()

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> tuple:
        """
        Returns:
            error_map: per-pixel reconstruction error at original frame size (H, W)
            mean_error: scalar mean error
        """
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.AE_SIZE, self.AE_SIZE))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        output = self.model(tensor)
        error = torch.abs(tensor - output).mean(dim=1).squeeze(0).cpu().numpy()
        mean_error = float(error.mean())
        error_map = cv2.resize(error, (w, h))
        return error_map, mean_error
