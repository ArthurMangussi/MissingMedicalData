"""
Ablation study for MAE-ViT / CMask-ViT, isolating the contribution of the
custom, missingness-aware patch masking strategy (algorithms/models_mae.py:
MaskedAutoencoderViT.custom_masking / forward_encoder_inpainting /
forward_inpainting) against alternative masking strategies.

Standard MAE (MaskedAutoencoderViT.forward, via random_masking) hides a
random subset of patches at a fixed ratio, blind to where the real
missing pixels are. The custom contribution instead derives the patch
mask from the true missing-pixel pattern, so only the actually-corrupted
patches are reconstructed. forward_inpainting takes that patch mask as a
plain tensor, so every strategy below reuses the exact same pretrained
weights, pre/post-processing, and pixel-level compositing (only real
missing pixels are ever replaced by the model's output -- observed
pixels are always copied through unchanged, regardless of which patches
the encoder was told to treat as masked). This isolates the effect of
the masking strategy itself from everything else in the pipeline.

Masking strategies compared
----------------------------
- any_pixel_proposed: a patch is masked if it contains at least one
  missing pixel -- the rule actually used in
  utils/MyModels.py::mae_imputer_transform.
- threshold_{25,50,75}: same informed principle (derived from the real
  missing-pixel pattern), but a patch is only masked if at least
  {25,50,75}% of its pixels are missing -- a sensitivity sweep on the
  patch-aggregation rule.
- random_matched: masks the SAME NUMBER of patches per image as
  any_pixel_proposed, but at uniformly random locations instead of the
  true missing-pixel locations -- isolates whether the benefit comes
  from the masking budget or from knowing WHERE the corruption is.
- vanilla_random_ratio_075: MAE's own native random masking at its
  ImageNet pretraining ratio (0.75), completely ignoring the real
  missingness pattern -- the "use MAE out of the box" baseline. Unlike
  the strategies above, this one is computed by calling the model's
  genuine, unmodified forward_encoder (which internally calls
  random_masking) + forward_decoder path directly -- not an emulation
  through custom_masking -- so it is literally the original MAE
  inference mechanism, not a re-implementation of its behavior.
- mixed_random_structured_25pct: the informed any_pixel_proposed mask,
  plus an additional uniformly-random 25% of the remaining (truly
  visible) patches also masked -- the same backbone evaluated with a mix
  of random and structured masking.

For every strategy, PSNR/SSIM/MAE are computed on the missing pixels
only (same convention as codes/experimental_design_*.py).
"""

import sys

sys.path.append("./")

import os

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from codes.data_amputation import ImageDataAmputation
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyModels import ModelsImputation

PATCH_SIZE = 16
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MECHANISMS = {
    "MCAR": lambda amp, x: amp.generate_mcar_dead_pixels(
        x, p_single=0.02, p_cluster=0.01, cluster_size=5
    ),
    "MNAR-SQUARES": lambda amp, x: amp.generate_squares_mask(x, square_size=30),
    "MNAR-LINES": lambda amp, x: amp.generate_stripes(x, frac_bad_cols=0.01, stripe_width=5),
}


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    """Normalize (N, H, W, 1) or (N, H, W) inputs down to (N, H, W)."""
    return np.squeeze(arr, axis=-1) if arr.ndim == 4 else arr


def pixel_mask_to_patch_fraction(missing_mask_np: np.ndarray, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Convert a pixel-level missing mask, (N, H, W) or (N, H, W, 1), into the
    fraction of missing pixels per patch, (N, num_patches). Every masking
    strategy below thresholds/samples from this same ground truth.
    """
    m = torch.from_numpy(_squeeze_channel(missing_mask_np)).float().unsqueeze(1)  # N, 1, H, W
    N, _, H, W = m.shape
    h, w = H // patch_size, W // patch_size
    m = m.reshape(N, 1, h, patch_size, w, patch_size)
    m = torch.einsum("nchpwq->nhwpqc", m)
    m = m.reshape(N, h * w, patch_size ** 2)
    return m.mean(dim=-1)


def mask_any_pixel(patch_fraction: torch.Tensor) -> torch.Tensor:
    return (patch_fraction > 0).float()


def mask_threshold(patch_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    return (patch_fraction >= threshold).float()


def mask_random_matched(patch_fraction: torch.Tensor) -> torch.Tensor:
    """Same per-image masked-patch COUNT as any_pixel_proposed, random location."""
    reference = mask_any_pixel(patch_fraction)
    N, L = reference.shape
    mask = torch.zeros_like(reference)
    for i in range(N):
        k = int(reference[i].sum().item())
        if k == 0:
            continue
        idx = torch.randperm(L)[:k]
        mask[i, idx] = 1.0
    return mask


def mask_mixed_random_structured(patch_fraction: torch.Tensor, extra_random_ratio: float = 0.25) -> torch.Tensor:
    """
    Structured (informed) mask over the truly corrupted patches, PLUS an
    additional uniformly-random extra_random_ratio of the remaining
    (truly visible) patches also masked. Models evaluating the same
    backbone with a mix of random and structured masking, as opposed to
    purely structured (any_pixel_proposed) or purely random
    (vanilla_random_ratio_075).
    """
    structured = mask_any_pixel(patch_fraction)
    N, L = structured.shape
    mixed = structured.clone()
    for i in range(N):
        visible_idx = (structured[i] == 0).nonzero(as_tuple=True)[0]
        k_extra = int(len(visible_idx) * extra_random_ratio)
        if k_extra == 0:
            continue
        extra_idx = visible_idx[torch.randperm(len(visible_idx))[:k_extra]]
        mixed[i, extra_idx] = 1.0
    return mixed


# Strategies that need an explicit informed/uninformed patch mask, routed
# through the custom masking path (model.forward_inpainting). The native
# vanilla-random-masking baseline is handled separately below, via the
# model's own unmodified forward_encoder/random_masking/forward_decoder --
# not an emulation through custom_masking -- since that is the literal
# "off-the-shelf MAE with standard random masking" comparison a reviewer
# can verify against the original MAE implementation.
MASKING_STRATEGIES = {
    "any_pixel_proposed": lambda pf: mask_any_pixel(pf),
    "threshold_25": lambda pf: mask_threshold(pf, 0.25),
    "threshold_50": lambda pf: mask_threshold(pf, 0.50),
    "random_matched": lambda pf: mask_random_matched(pf),
}

VANILLA_RANDOM_RATIO = 0.75
VANILLA_STRATEGY_NAME = "vanilla_random_ratio_075"


def _preprocess(x_test_md_np: np.ndarray, device: str):
    """Shared ImageNet-normalization preprocessing used by every MAE-ViT variant."""
    if x_test_md_np.ndim == 3:
        x_test_md_np = np.expand_dims(x_test_md_np, axis=-1)

    x_limpo = np.nan_to_num(x_test_md_np, nan=0.0)
    x = torch.from_numpy(x_limpo).float().to(device)
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 1, 3)
    x = torch.einsum("nhwc->nchw", x)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    x = (x - mean) / std
    return x, x_limpo, mean, std


def _composite(x_limpo: np.ndarray, y_recon_np: np.ndarray, missing_mask_test_np: np.ndarray) -> np.ndarray:
    m_np = missing_mask_test_np
    if m_np.ndim == 3:
        m_np = m_np[..., np.newaxis]
    return x_limpo * (1 - m_np) + y_recon_np[..., :1] * m_np


@torch.no_grad()
def mae_ablation_transform(
    model, x_test_md_np: np.ndarray, missing_mask_test_np: np.ndarray, m_patch: torch.Tensor, device: str
) -> np.ndarray:
    """
    Same preprocessing/compositing as
    utils.MyModels.ModelsImputation.mae_imputer_transform, but takes a
    precomputed patch-level mask so different masking strategies can be
    compared with everything else held fixed.
    """
    model.eval()
    x, x_limpo, mean, std = _preprocess(x_test_md_np, device)

    pred_patches = model.forward_inpainting(x, m_patch.to(device))
    y_recon = model.unpatchify(pred_patches)
    y_recon = y_recon * std + mean
    y_recon_np = torch.einsum("nchw->nhwc", y_recon).cpu().numpy()

    return _composite(x_limpo, y_recon_np, missing_mask_test_np)


@torch.no_grad()
def mae_vanilla_random_transform(
    model, x_test_md_np: np.ndarray, missing_mask_test_np: np.ndarray, mask_ratio: float, device: str
) -> np.ndarray:
    """
    The genuine, unmodified MAE inference path -- model.forward_encoder
    (which calls random_masking internally) followed by
    model.forward_decoder, i.e. model.forward() minus the training loss.
    This is the literal off-the-shelf MAE with standard random masking,
    not a re-implementation of its behavior via custom_masking.
    """
    model.eval()
    x, x_limpo, mean, std = _preprocess(x_test_md_np, device)

    latent, _, ids_restore = model.forward_encoder(x, mask_ratio)
    pred_patches = model.forward_decoder(latent, ids_restore)
    y_recon = model.unpatchify(pred_patches)
    y_recon = y_recon * std + mean
    y_recon_np = torch.einsum("nchw->nhwc", y_recon).cpu().numpy()

    return _composite(x_limpo, y_recon_np, missing_mask_test_np)


def compute_metrics(x_test: np.ndarray, x_test_imputed: np.ndarray, missing_mask_test: np.ndarray) -> dict:
    """PSNR/SSIM/MAE computed on the missing pixels only (matches experimental_design_*.py)."""
    x_test = _squeeze_channel(x_test)
    x_test_imputed = _squeeze_channel(x_test_imputed)
    missing_mask_test_binary = _squeeze_channel(missing_mask_test).astype(bool)

    x_imputed_missing = x_test_imputed[missing_mask_test_binary]
    x_original_missing = x_test[missing_mask_test_binary]

    mae = mean_absolute_error(x_imputed_missing, x_original_missing)
    psnr = peak_signal_noise_ratio(x_imputed_missing, x_original_missing, data_range=1.0)
    ssim = structural_similarity(x_imputed_missing, x_original_missing, data_range=1.0)

    return {"MAE": round(float(mae), 4), "PSNR": round(float(psnr), 4), "SSIM": round(float(ssim), 4)}


def load_baseline_data(dataset_name: str):
    """Same array order used across codes/experimental_design_*.py and classification_vgg16.py."""
    data = Datasets(dataset_name)

    if dataset_name == "cbis-ddsm":
        images, _, labels = data.load_data()
        return images, np.array(labels)

    images, y_dict, image_ids = data.load_data()
    if dataset_name == "inbreast":
        labels = np.array([y_dict[i] for i in image_ids])
    else:
        labels = np.array(list(y_dict.values()))

    return images, labels


def run_ablation(
    dataset_name: str,
    mechanisms: list,
    device: str = "cpu",
    n_folds: int = 5,
    checkpoint_path: str = "/home/gpu-10-2025/Área de trabalho/Modelos/mae_visualize_vit_large.pth",
    output_dir: str = "./results/ablation_mae_vit",
) -> pd.DataFrame:
    _logger = MeLogger()
    os.makedirs(output_dir, exist_ok=True)

    images, labels = load_baseline_data(dataset_name)

    model = ModelsImputation.model_mae_vit(checkpoint_path).to(device)
    model.eval()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    rows = []
    for mechanism in mechanisms:
        amputation = ImageDataAmputation()

        for fold, (_, test_idx) in enumerate(skf.split(images, labels)):
            _logger.info(
                f"[MAE-ViT masking ablation][{dataset_name}][{mechanism}] Fold {fold + 1}/{n_folds}"
            )

            x_test = images[test_idx]
            x_test, x_test_md, missing_mask_test = MECHANISMS[mechanism](amputation, x_test)
            patch_fraction = pixel_mask_to_patch_fraction(missing_mask_test)

            # Native vanilla-random-masking baseline: model's own unmodified
            # forward_encoder/random_masking + forward_decoder path.
            x_test_imputed = mae_vanilla_random_transform(
                model, x_test_md, missing_mask_test, VANILLA_RANDOM_RATIO, device
            )
            metrics = compute_metrics(x_test, x_test_imputed, missing_mask_test)
            rows.append(
                {
                    "DATASET": dataset_name,
                    "MECHANISM": mechanism,
                    "FOLD": fold,
                    "MASKING_STRATEGY": VANILLA_STRATEGY_NAME,
                    "MASKED_PATCH_FRACTION_MEAN": VANILLA_RANDOM_RATIO,
                    **metrics,
                }
            )

            # Informed/uninformed masking strategies, all routed through the
            # custom masking path (model.forward_inpainting) so only the
            # patch mask itself differs between conditions.
            for strategy_name, strategy_fn in MASKING_STRATEGIES.items():
                m_patch = strategy_fn(patch_fraction)

                x_test_imputed = mae_ablation_transform(
                    model, x_test_md, missing_mask_test, m_patch, device
                )
                metrics = compute_metrics(x_test, x_test_imputed, missing_mask_test)

                rows.append(
                    {
                        "DATASET": dataset_name,
                        "MECHANISM": mechanism,
                        "FOLD": fold,
                        "MASKING_STRATEGY": strategy_name,
                        "MASKED_PATCH_FRACTION_MEAN": round(m_patch.mean().item(), 4),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_mae_vit_masking_ablation.csv"), index=False
    )
    return df


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ablation is run on one dataset across all three missingness mechanisms
    # by default (already 5 folds x 3 mechanisms x 6 strategies = 90 forward
    # passes through a ViT-Large); add more dataset names to widen coverage.
    dataset_names = ["inbreast", "mias", "vindr-reduzido", "cbis-ddsm"]
    mechanisms = ["MCAR", "MNAR-SQUARES", "MNAR-LINES"]

    for dataset_name in dataset_names:
        run_ablation(dataset_name, mechanisms, device=DEVICE)
