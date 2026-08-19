"""
Downstream classification task (baseline vs. imputed images) using a
VGG16 backbone with transfer learning.

For each dataset, one VGG16-based classifier is trained per cross-validation
fold using only the original (complete) mammography images -- the very same
StratifiedKFold split (n_splits=5, random_state=42) used in
codes/experimental_design_*.py, so the held-out test set of a fold matches
the images that were amputated and later imputed by the inpainting methods.

That classifier is then evaluated twice per fold:
    1. On the clean held-out test images -> baseline performance.
    2. On the imputed version of the same test images (loaded from
       ./new_results/{dataset}/{model_impt}/imputed_images/fold{fold}_{mechanism}/)
       -> downstream performance after inpainting.

Comparing both tells how much diagnostic classification performance is
preserved (or lost) by each imputation method/mechanism.
"""

import sys

sys.path.append("./")

import gc
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

from utils.MeLogSingle import MeLogger
from utils.MyDataset import CustomImageDataset, Datasets

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CustomImageDataset repeats the grayscale channel 3x whenever the last
# transform is a Normalize, which is exactly what VGG16 (ImageNet-pretrained)
# expects as input.
TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def build_vgg16(
    num_classes: int = 2, freeze_backbone: bool = True, device: str = "cpu"
) -> nn.Module:
    """
    Build a VGG16 (ImageNet-pretrained) backbone for a binary classification
    downstream task, replacing the last classifier layer.

    freeze_backbone=True freezes the ENTIRE pretrained network (conv features
    plus the two 4096-unit FC layers) so only the newly created output layer
    trains -- a true linear probe. Freezing just model.features would leave
    ~119M FC parameters trainable, which massively overfits the few hundred
    training images typical of a single cross-validation fold here.
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model.to(device)


def train_vgg16(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 10,
) -> nn.Module:
    """Fine-tune the classifier head of a VGG16 backbone with early stopping."""
    train_loader = DataLoader(
        CustomImageDataset(x_train, y_train, transform=TRANSFORM),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CustomImageDataset(x_val, y_val, transform=TRANSFORM),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def evaluate_vgg16(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Compute classification metrics of a trained VGG16 model on a test set."""
    test_loader = DataLoader(
        CustomImageDataset(x_test, y_test, transform=TRANSFORM),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    y_true, y_pred, y_score = [], [], []
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

    metrics = {
        "ACCURACY": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PRECISION": precision_score(y_true, y_pred, zero_division=0),
        "RECALL": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["AUC"] = np.nan

    return metrics


def load_baseline_data(dataset_name: str):
    """
    Load raw uint8 [0, 255] images, integer labels and (when available)
    patient ids for a dataset, in the exact array order used by
    codes/experimental_design_*.py when generating the imputed images --
    so the folds recomputed here line up with the saved results.

    patients is None for every dataset except breakhist, which signals
    run_classification to use the same patient-grouped split
    (StratifiedGroupKFold) as codes/experimental_design_breakhist.py instead
    of a plain StratifiedKFold.
    """
    data = Datasets(dataset_name)

    if dataset_name == "breakhist":
        images, _, labels, patients = data.load_data()
        return images, np.array(labels), np.array(patients)

    if dataset_name == "cbis-ddsm":
        images, _, labels = data.load_data()
        return images, np.array(labels), None

    images, y_dict, image_ids = data.load_data()

    if dataset_name == "inbreast":
        labels = np.array([y_dict[i] for i in image_ids])
    else:
        labels = np.array(list(y_dict.values()))

    return images, labels, None


def load_imputed_fold_images(
    dataset_name: str,
    model_impt: str,
    mechanism: str,
    fold: int,
    n_images: int,
    results_dir: str = "./new_results",
):
    """
    Load the imputed test images saved by codes/experimental_design_*.py for
    a given fold, in the exact order they were written (IMG_0000.png, ...).

    Also returns the ground-truth labels for those images when they can be
    read from the classes.csv saved alongside the PNGs (currently only
    utils.MyUtils.Utilities.save_image_breakhist writes one keyed by the
    "IMG_XXXX" id -- save_image/save_image_cbis don't, so labels is None for
    every other dataset and the caller falls back to the y_test it already
    has from its own fold split). Reading the label from classes.csv instead
    of trusting that a freshly recomputed split lines up positionally with
    what was saved is what actually makes this robust to that split ever
    drifting (sklearn/numpy version differences, etc.) -- IMG_XXXX.png's
    array position alone doesn't guarantee that.

    Returns (None, None) if this dataset/imputer/mechanism/fold combination
    was not found on disk.
    """
    fold_dir = os.path.join(
        results_dir, dataset_name, model_impt, "imputed_images", f"fold{fold}_{mechanism}"
    )
    if not os.path.isdir(fold_dir):
        return None, None

    classes_path = os.path.join(fold_dir, "classes.csv")
    labels_by_image = None
    if os.path.exists(classes_path):
        classes_df = pd.read_csv(classes_path)
        if "Image" in classes_df.columns and "Target" in classes_df.columns:
            labels_by_image = dict(zip(classes_df["Image"], classes_df["Target"]))

    images = []
    labels = []
    for i in range(n_images):
        image_id = f"IMG_{i:04d}"
        img_path = os.path.join(fold_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            return None, None
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        images.append(img)

        if labels_by_image is not None:
            if image_id not in labels_by_image:
                return None, None
            labels.append(int(labels_by_image[image_id]))

    return np.array(images), (np.array(labels) if labels_by_image is not None else None)


def run_classification(
    dataset_name: str,
    mechanisms: list,
    imputers: list,
    device: str = "cpu",
    freeze_backbone: bool = True,
    results_dir: str = "./new_results",
    output_dir: str = "./results/classification",
):
    _logger = MeLogger()
    os.makedirs(output_dir, exist_ok=True)

    images, labels, patients = load_baseline_data(dataset_name)

    # breakhist has multiple crops per patient, so codes/experimental_design_breakhist.py
    # splits by patient (StratifiedGroupKFold) to avoid leaking a patient's crops
    # across train/val/test -- reusing a plain StratifiedKFold here would desync
    # the test-fold indices from the images already saved under new_results/breakhist/.
    if patients is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(images, labels, groups=patients)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(images, labels)

    baseline_rows = {}
    imputed_rows = {
        (mechanism, model_impt): {} for mechanism in mechanisms for model_impt in imputers
    }

    for fold, (train_val_idx, test_idx) in enumerate(split_iter):
        _logger.info(f"[VGG16 classification][{dataset_name}] Fold {fold + 1}/5")

        x_train_val, x_test = images[train_val_idx], images[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        if patients is not None:
            patients_train_val = patients[train_val_idx]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
            train_idx, val_idx = next(
                gss.split(x_train_val, y_train_val, groups=patients_train_val)
            )
            x_train, x_val = x_train_val[train_idx], x_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                test_size=0.2,
                random_state=fold,
                stratify=y_train_val,
            )

        model = build_vgg16(num_classes=2, freeze_backbone=freeze_backbone, device=device)
        model = train_vgg16(model, x_train, y_train, x_val, y_val, device=device)

        baseline_rows[f"fold{fold}"] = evaluate_vgg16(model, x_test, y_test, device=device)

        for mechanism in mechanisms:
            for model_impt in imputers:
                x_test_imputed, y_test_saved = load_imputed_fold_images(
                    dataset_name,
                    model_impt,
                    mechanism,
                    fold,
                    n_images=len(test_idx),
                    results_dir=results_dir,
                )
                if x_test_imputed is None:
                    _logger.info(
                        f"[VGG16 classification][{dataset_name}] "
                        f"Skipping {model_impt}/{mechanism} fold{fold}: results not found"
                    )
                    continue

                # Prefer the labels saved alongside the PNGs (classes.csv) over
                # the freshly recomputed y_test -- they're the actual record of
                # what each IMG_XXXX.png is. Falls back to y_test only for the
                # older save_image/save_image_cbis outputs that don't have one.
                if y_test_saved is not None:
                    if not np.array_equal(y_test_saved, y_test):
                        _logger.error(
                            f"[VGG16 classification][{dataset_name}] "
                            f"{model_impt}/{mechanism} fold{fold}: classes.csv labels "
                            "don't match the recomputed fold split -- the saved images "
                            "no longer correspond to this test fold, skipping"
                        )
                        continue
                    y_test_eval = y_test_saved
                else:
                    y_test_eval = y_test

                imputed_rows[(mechanism, model_impt)][f"fold{fold}"] = evaluate_vgg16(
                    model, x_test_imputed, y_test_eval, device=device
                )

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(baseline_rows).T.to_csv(
        os.path.join(output_dir, f"{dataset_name}_baseline_vgg16_classification.csv")
    )

    for (mechanism, model_impt), rows in imputed_rows.items():
        if not rows:
            continue
        pd.DataFrame(rows).T.to_csv(
            os.path.join(
                output_dir, f"{dataset_name}_{model_impt}_{mechanism}_vgg16_classification.csv"
            )
        )


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_mechanisms = {
        #"inbreast": ["MNAR-SQUARES", "MNAR-LINES"],
        #"mias": ["MNAR-SQUARES", "MNAR-LINES"],
        #"vindr-reduzido": ["MNAR-SQUARES", "MNAR-LINES"],
        #"cbis-ddsm": ["MCAR"],
        "breakhist": ["MCAR", "MAR", "MNAR"],
    }
    imputers = ["knn", "dip", "mat", "harp", "mae-vit"]

    for dataset_name, mechanisms in dataset_mechanisms.items():
        run_classification(
            dataset_name=dataset_name,
            mechanisms=mechanisms,
            imputers=imputers,
            device=DEVICE,
        )
