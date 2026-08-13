"""Inference wrapper around HARP's restoration model (a Palette-style
conditional DDPM), adapted from harp/pipeline/artifact_restoration.py's
RestorationModel.inpaint() to this project's imputer convention:
`.transform(x_md, missing_mask)` with `missing_mask == 1` marking a missing
pixel, `== 0` an observed one. HARP's own mask convention already matches
this (`mask == 1` is the hole to be diffused, see network.py::restoration()'s
`y_t = y_0*(1.-mask) + mask*y_t`), so unlike MATInpainter no inversion is
needed here.

Architecture kwargs below (unet/beta_schedule) come from the upstream repo's
config/config_restoration_model.json (not shipped in the HARPipe PyPI
package) - they must match whatever checkpoint is loaded, since HARP's
Network takes them as plain constructor args rather than embedding them in
the checkpoint (unlike MAT's torch_utils.persistence approach). The values
here match the released `restoration_model.pth`, trained on the BCSS
histopathology dataset at 256x256 - a reasonable starting point for BreaKHis
(also H&E breast histopathology) either used directly or fine-tuned.
"""

import cv2
import numpy as np
import torch

from algorithms.harp.network import Network

UNET_KWARGS = dict(
    in_channel=6,
    out_channel=3,
    inner_channel=64,
    channel_mults=[1, 2, 4, 8],
    attn_res=[16],
    num_head_channels=32,
    res_blocks=2,
    dropout=0.2,
    image_size=256,
)
BETA_SCHEDULE = {
    "train": dict(schedule="linear", n_timestep=1000, linear_start=1e-6, linear_end=0.01),
    "test": dict(schedule="linear", n_timestep=250, linear_start=1e-4, linear_end=0.09),
}


class HARPInpainter:
    """Runs HARP's diffusion restoration model on batches of (possibly grayscale) images."""

    def __init__(
        self,
        checkpoint_path: str,
        resolution: int = 256,
        device: str = "cuda",
        unet_kwargs: dict = None,
        beta_schedule: dict = None,
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.resolution = resolution

        self.model = Network(
            unet=unet_kwargs or UNET_KWARGS,
            beta_schedule=beta_schedule or BETA_SCHEDULE,
            module_name="guided_diffusion",
            init_type="kaiming",
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.set_new_noise_schedule(device=self.device, phase="test")
        self.model.eval()

    def _prepare(self, image: np.ndarray, mask: np.ndarray):
        """image (H,W)[,1] any range + mask (H,W)[,1] 1=missing/0=known
        -> (3,res,res) image tensor in [-1,1], (1,res,res) mask tensor (1=hole), orig (H,W)."""
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        # Missing pixels arrive as NaN in this project's convention (see
        # ModelsImputation.mae_imputer_transform); the hole area of the image
        # is discarded below via the mask anyway, so filling with 0 is fine.
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        orig_shape = arr.shape[:2]
        if orig_shape != (self.resolution, self.resolution):
            arr = cv2.resize(arr, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        rgb_chw = np.repeat(arr[None, ...], 3, axis=0)  # (3,res,res)
        img = (rgb_chw * 2.0 - 1.0).astype(np.float32)  # [-1,1]

        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.max() > 1.0 + 1e-6:
            m = m / 255.0
        if m.shape[:2] != (self.resolution, self.resolution):
            m = cv2.resize(m, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        mask_arr = (m > 0.5).astype(np.float32)[None, ...]  # (1,res,res), 1=hole

        return img, mask_arr, orig_shape

    def transform(self, x_md_batch: np.ndarray, missing_mask_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_md_batch : np.ndarray, shape (N,H,W) or (N,H,W,1)
            Grayscale images with missing pixels (NaN or any fill value in the
            holes - discarded before diffusion).
        missing_mask_batch : np.ndarray, same leading shape as x_md_batch
            1 = missing pixel, 0 = observed pixel.

        Returns
        -------
        np.ndarray
            Imputed images, same shape and [0, 1] value range as x_md_batch.
        """
        keep_channel_dim = x_md_batch.ndim == 4 and x_md_batch.shape[-1] == 1
        out = np.empty(x_md_batch.shape, dtype=np.float32)

        for i in range(x_md_batch.shape[0]):
            image_chw, mask_hw, orig_shape = self._prepare(x_md_batch[i], missing_mask_batch[i])

            image_t = torch.from_numpy(image_chw).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask_hw).unsqueeze(0).to(self.device)

            # Matches RestorationModel.inpaint(): known area keeps its real
            # value, the hole starts from Gaussian noise (not the corrupted
            # input), which is what this checkpoint was trained to denoise.
            cond_image = image_t * (1.0 - mask_t) + mask_t * torch.randn_like(image_t)

            with torch.no_grad():
                y_t, _ = self.model.restoration(cond_image, y_t=cond_image, y_0=image_t, mask=mask_t)

            output = (y_t[0].clamp(-1, 1) + 1.0) / 2.0  # (3,res,res) in [0,1]
            output_np = output.mean(dim=0).cpu().numpy()  # collapse back to grayscale

            if output_np.shape != orig_shape:
                output_np = cv2.resize(output_np, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

            out[i] = output_np[..., None] if keep_channel_dim else output_np

        return out
