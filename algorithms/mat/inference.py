"""Inference wrapper around a trained MAT (Mask-Aware Transformer) generator.

Loads a checkpoint produced either by the official MAT training code or by
`codes/train_mat.py` (both save the same `network-snapshot-*.pkl` format, see
`training/training_loop.py`) and exposes a `.transform(x_md, missing_mask)`
method that follows the convention used by the other imputers wired into
`utils/MyModels.py`: `missing_mask == 1` marks a missing pixel, `== 0` an
observed one. MAT itself uses the opposite convention internally
(`masks_in == 1` means "keep this pixel", `== 0` means "hole"), so this
wrapper inverts the mask before calling the generator.
"""

import cv2
import numpy as np
import torch

import algorithms.mat  # noqa: F401  (adds the vendored MAT repo root to sys.path)
import dnnlib
import legacy


class MATInpainter:
    """Runs MAT inference on batches of (possibly grayscale) images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode

        # `G_ema` unpickles as a ready-to-use nn.Module (torch_utils/persistence.py
        # embeds the class source), so it can be used as-is instead of being rebuilt
        # from Generator(...) defaults and weight-copied like generate_image.py does
        # upstream (that dance exists there to bridge older/converted checkpoints
        # onto today's class definition; checkpoints from codes/train_mat.py already
        # come from this exact code, and rebuilding with default kwargs silently
        # mismatches whatever channel_base/mapping-layer count the "auto" cfg picked
        # at train time).
        with dnnlib.util.open_url(checkpoint_path) as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].to(self.device).eval().requires_grad_(False)
        self.resolution = self.G.img_resolution
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)

    def _prepare_image(self, image: np.ndarray):
        """(H,W)[,1] any range -> (3,res,res) float32 in [-1,1], plus the original (H,W)."""
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        # x_md_batch carries NaN under the missing-pixel hole (ImageDataAmputation
        # convention). MAT's own forward pass computes images_in * masks_in to zero
        # the hole out, but NaN * 0 stays NaN in IEEE floats, so an unfilled NaN
        # poisons the whole generator forward pass instead of being ignored. Fill it
        # before any resize/rescale so it can't spread further via interpolation.
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        orig_shape = arr.shape[:2]
        if orig_shape != (self.resolution, self.resolution):
            arr = cv2.resize(arr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        rgb_chw = np.repeat(arr[None, ...], 3, axis=0)  # (3,res,res)
        return (rgb_chw * 2.0 - 1.0).astype(np.float32), orig_shape

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        """(H,W)[,1], 1=missing/0=known -> (1,res,res) float32, 1=known/0=hole (MAT convention)."""
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        if arr.shape[:2] != (self.resolution, self.resolution):
            arr = cv2.resize(arr, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        known = 1.0 - (arr > 0.5).astype(np.float32)
        return known[None, ...]

    def transform(self, x_md_batch: np.ndarray, missing_mask_batch: np.ndarray, seed: int = 240) -> np.ndarray:
        """
        Parameters
        ----------
        x_md_batch : np.ndarray, shape (N,H,W) or (N,H,W,1)
            Grayscale images with missing pixels (the fill value inside the holes is
            irrelevant, MAT zeroes them out internally before encoding).
        missing_mask_batch : np.ndarray, same leading shape as x_md_batch
            1 = missing pixel, 0 = observed pixel.
        seed : int, optional
            Seed for the per-image latent noise `z`, kept fixed by default for
            reproducible experiment runs.

        Returns
        -------
        np.ndarray
            Imputed images, same shape and [0, 1] value range as x_md_batch.
        """
        keep_channel_dim = x_md_batch.ndim == 4 and x_md_batch.shape[-1] == 1
        out = np.empty(x_md_batch.shape, dtype=np.float32)
        rng = np.random.RandomState(seed)

        with torch.no_grad():
            for i in range(x_md_batch.shape[0]):
                image_chw, orig_shape = self._prepare_image(x_md_batch[i])
                mask_hw = self._prepare_mask(missing_mask_batch[i])

                image_t = torch.from_numpy(image_chw).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask_hw).unsqueeze(0).to(self.device)
                z = torch.from_numpy(rng.randn(1, self.G.z_dim).astype(np.float32)).to(self.device)

                output = self.G(
                    image_t, mask_t, z, self.label,
                    truncation_psi=self.truncation_psi, noise_mode=self.noise_mode,
                )
                output = (output[0].clamp(-1, 1) + 1.0) / 2.0  # (3,res,res) in [0,1]
                output_np = output.mean(dim=0).cpu().numpy()  # collapse back to grayscale

                if output_np.shape != orig_shape:
                    output_np = cv2.resize(output_np, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

                out[i] = output_np[..., None] if keep_channel_dim else output_np

        return out
