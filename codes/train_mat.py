"""
Train MAT (Mask-Aware Transformer for Large Hole Image Inpainting, Li et al.,
CVPR 2022) on one of this project's mammography datasets.

No checkpoint pretrained on mammography exists for MAT (the released weights
are CelebA-HQ / FFHQ / Places365), so it has to be trained from scratch here
before it can be used as an imputer. This script exports one of the project's
in-memory datasets (see utils/MyDataset.py) to a folder of PNGs and drives the
vendored training loop in algorithms/mat/ (256x256 config, since the
mammography crops used across this project are 224x224).

The resulting `network-snapshot-*.pkl` (same format `legacy.load_network_pkl`
reads) is what `utils/MyModels.py::ModelsImputation.model_mat` loads via
`algorithms.mat.inference.MATInpainter` for the "mat" entry in
`ModelsImputation.choose_model`, used from the codes/experimental_design_*.py
scripts.

License note: MAT's code (algorithms/mat/) is released under NVIDIA's
Source Code License-NC - research/non-commercial use only.

Usage
-----
    python codes/train_mat.py --dataset cbis-ddsm --outdir ./training-runs/mat --gpus 1
"""

import argparse
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.append("./")

from algorithms.mat.train import UserError, setup_training_loop_kwargs, subprocess_fn
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets

DATALOADER = "datasets.dataset_256.ImageFolderMaskDataset"  # matches algorithms/mat/datasets/dataset_256.py


def _export_image_folder(images: np.ndarray, out_dir: str):
    """Writes a batch of (N,H,W) grayscale images as PNGs for MAT's ImageFolderMaskDataset."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), img)


def train_mat(
    dataset_name: str,
    outdir: str,
    gpus: int = 1,
    kimg: int = None,
    snap: int = 10,
    val_size: float = 0.1,
    resume: str = None,
    batch: int = None,
):
    logger = MeLogger()

    data = Datasets(dataset_name)
    images, *_ = data.load_data()  # (N, 224, 224) grayscale images
    images_train, images_val = train_test_split(images, test_size=val_size, random_state=42)

    workdir = tempfile.mkdtemp(prefix=f"mat_{dataset_name}_")
    train_dir = os.path.join(workdir, "train")
    val_dir = os.path.join(workdir, "val")
    logger.info(
        f"[MAT] Exporting {len(images_train)} train / {len(images_val)} val images to {workdir}"
    )
    _export_image_folder(images_train, train_dir)
    _export_image_folder(images_val, val_dir)

    try:
        run_desc, args = setup_training_loop_kwargs(
            gpus=gpus,
            snap=snap,
            metrics=[],
            data=train_dir,
            data_val=val_dir,
            dataloader=DATALOADER,
            cfg="auto",
            kimg=kimg,
            resume=resume,
            batch=batch,
        )
    except UserError as err:
        shutil.rmtree(workdir, ignore_errors=True)
        logger.error(f"[MAT] {err}")
        raise

    os.makedirs(outdir, exist_ok=True)
    prev_run_dirs = [d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]
    prev_ids = [int(d[:5]) for d in prev_run_dirs if d[:5].isdigit()]
    cur_id = max(prev_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f"{cur_id:05d}-{run_desc}")
    os.makedirs(args.run_dir)

    logger.info(f"[MAT] Training run: {args.run_dir}")
    torch.multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

    shutil.rmtree(workdir, ignore_errors=True)
    logger.info(f"[MAT] Done. Checkpoints saved under {args.run_dir}")
    return args.run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAT on a mammography dataset")
    parser.add_argument(
        "--dataset", required=True, choices=["cbis-ddsm", "inbreast", "mias", "vindr-reduzido"]
    )
    parser.add_argument("--outdir", default="./training-runs/mat")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--kimg", type=int, default=None, help="Override training duration (thousands of images)"
    )
    parser.add_argument("--snap", type=int, default=10, help="Snapshot interval, in ticks")
    parser.add_argument("--resume", default=None, help="Path/URL of a checkpoint to resume from")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "Override total batch size (must be divisible by --gpus). "
            "The 'auto' config picks 16 at 256x256, which needs more than ~8GB of VRAM; "
            "lower this (e.g. 4 or 2) on smaller GPUs to avoid CUDA OOM."
        ),
    )
    cli_args = parser.parse_args()

    train_mat(
        dataset_name=cli_args.dataset,
        outdir=cli_args.outdir,
        gpus=cli_args.gpus,
        kimg=cli_args.kimg,
        snap=cli_args.snap,
        resume=cli_args.resume,
        batch=cli_args.batch,
    )
