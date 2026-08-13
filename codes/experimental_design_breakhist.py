"""
Main code for run Experimental Setup for Missing Data
Imputation into Images (i.e., Image Inpainting) — BreaKHis dataset
(histopatologia de mama, benigno vs. maligno).

Diferente dos demais experimental_design_*.py, aqui a divisão em folds usa
StratifiedGroupKFold/GroupShuffleSplit agrupados por paciente: o BreaKHis tem
vários recortes por lâmina/paciente, então dividir por imagem deixaria
recortes do mesmo paciente vazarem entre treino/validação/teste.
"""
import sys

sys.path.append("./")
import gc

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from time import perf_counter

from codes.data_amputation import ImageDataAmputation
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyModels import ModelsImputation
from utils.MyUtils import Utilities

BATCH_SIZE = 16  # Ajuste conforme a memória da sua GPU


def run_experimental_design(
    model_impt: str,
    md_mechanism: str,
    images: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    patients: np.ndarray,
    dataset_name: str,
    batch_size: int = BATCH_SIZE,
):
    os.makedirs(f"./results/{model_impt}", exist_ok=True)
    _logger = MeLogger()
    ut = Utilities()
    results_mse = {}
    results_psnr = {}
    results_ssim = {}

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(
        sgkf.split(images, labels, groups=patients)
    ):
        _logger.info(f"\n[Fold {fold + 1}/5] - {dataset_name}")

        x_train_val = images[train_val_idx]
        y_train_val = labels[train_val_idx]
        patients_train_val = patients[train_val_idx]

        x_test = images[test_idx]
        img_test_filenames = filenames[test_idx]
        img_test_labels = labels[test_idx]
        img_test_patients = patients[test_idx]

        # Divide treino e validação internamente por paciente (20% para validação)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
        train_idx, val_idx = next(
            gss.split(x_train_val, y_train_val, groups=patients_train_val)
        )
        x_train, x_val = x_train_val[train_idx], x_train_val[val_idx]

        amputation = ImageDataAmputation()

        x_train, x_train_md, _ = amputation.generate_mcar_dead_pixels(
            x_train, p_single=0.0, p_cluster=0.3, cluster_size=25
        )
        x_val, x_val_md, _ = amputation.generate_mcar_dead_pixels(
            x_val, p_single=0.00, p_cluster=0.3, cluster_size=25
        )
        x_test, x_test_md, missing_mask_test = amputation.generate_mcar_dead_pixels(
            x_test, p_single=0.00, p_cluster=0.3, cluster_size=25
        )

        model = ModelsImputation()
        imputer = model.choose_model(
            model=model_impt,
            x_train=x_train,
            x_val_md=x_val_md,
            x_val=x_val,
            x_train_md=x_train_md,
        )

        # Processa o fold de teste em lotes de `batch_size` imagens: um fold
        # inteiro (centenas/milhares de imagens 224x224) de uma só vez
        # estoura a memória da GPU em modelos como MAE-ViT/MAT/diffusion, que
        # carregam o lote inteiro no device. Cada lote é processado e
        # concatenado de volta ao final, preservando o fold completo nas
        # métricas e no salvamento.
        n_test = len(x_test_md)
        n_batches = (n_test + batch_size - 1) // batch_size
        x_test_imputed_chunks = []

        for b in range(n_batches):
            start, end = b * batch_size, min((b + 1) * batch_size, n_test)
            batch_x_md = x_test_md[start:end]
            batch_mask = missing_mask_test[start:end]

            _logger.info(
                f"[{model_impt}] Lote {b + 1}/{n_batches} ({end - start} imagens)"
            )

            if model_impt == "mae-vit" or model_impt == "mae-vit-gan":
                # MAE-ViT expects incomplete image for inpainting
                batch_imputed = model.mae_imputer_transform(
                    model=imputer,
                    x_test_md_np=batch_x_md,
                    missing_mask_test_np=batch_mask,
                )

            elif model_impt in ("mc", "mat"):
                batch_imputed = imputer.transform(batch_x_md, batch_mask)

            elif model_impt == "diffusion":
                # Diffusion model - use incomplete image
                prompt = "Breast histopathology, H&E-stained tissue slide, high-quality, no artifacts, inpainting task."
                batch_imputed = model.diffusion_transform(
                    model=imputer,
                    x_test_md_np=np.squeeze(batch_x_md, axis=-1),
                    missing_mask_test_np=np.squeeze(batch_mask, axis=-1),
                    prompt=prompt,
                    num_inference_steps=150,
                )

            elif model_impt == "dip":
                batch_x_2d = batch_x_md.squeeze(-1)
                batch_x_torch = np.expand_dims(batch_x_2d, axis=-1).transpose(0, 3, 1, 2)
                batch_mask_4d = (
                    batch_mask if batch_mask.ndim == 4 else np.expand_dims(batch_mask, axis=-1)
                )
                batch_mask_torch = batch_mask_4d.transpose(0, 3, 1, 2)

                imputed_batch = imputer.fit_and_transform(batch_x_torch, batch_mask_torch)
                batch_imputed = imputed_batch.transpose(0, 2, 3, 1)

            else:
                # KNN, MICE, etc.
                batch_imputed = imputer.transform(batch_x_md)

            x_test_imputed_chunks.append(batch_imputed)

        x_test_imputed = np.concatenate(x_test_imputed_chunks, axis=0)

        ## Save the reconstructed image, keeping full traceability (filename,
        ## patient e label) para uma tarefa downstream de classificação futura
        ut.save_image_breakhist(
            mechanism=md_mechanism,
            images=x_test_imputed,
            missing_masks=missing_mask_test,
            fold=fold,
            model_impt=model_impt,
            dataset=dataset_name,
            image_filenames=img_test_filenames,
            image_labels=img_test_labels,
            image_patients=img_test_patients,
        )

        ## Measure the imputation performance
        # generate_mcar_dead_pixels always returns 4D arrays (N, H, W, 1) for
        # this single-channel dataset, so x_test, x_test_imputed and the mask
        # must all be squeezed the same way before boolean-indexing -- x_test
        # being left 4D here previously made x_original_missing keep a
        # trailing (..., 1) axis that x_imputed_missing didn't have, which is
        # what made PSNR/SSIM's shape-equality check fail.
        missing_mask_test_binary = missing_mask_test.astype(bool)
        if x_test.ndim == 4:
            x_test = np.squeeze(x_test, axis=-1)
        if x_test_imputed.ndim == 4:
            x_test_imputed = np.squeeze(x_test_imputed, axis=-1)
        if missing_mask_test_binary.ndim == 4:
            missing_mask_test_binary = np.squeeze(missing_mask_test_binary, axis=-1)

        # Extract only missing pixels from both images
        x_imputed_missing = x_test_imputed[missing_mask_test_binary]
        x_original_missing = x_test[missing_mask_test_binary]

        # BreaKHis images are single-channel grayscale (N, H, W) -- there is
        # no channel axis to loop over here, so MAE/PSNR/SSIM are computed
        # once, over every missing pixel in the whole fold (same convention
        # as codes/ablation.py::compute_metrics). A per-"channel" loop over
        # x_test.shape[0] would actually loop over images, not channels, and
        # can crash structural_similarity when a single image happens to
        # have fewer missing pixels than skimage's default win_size (7).
        mae = mean_absolute_error(x_imputed_missing, x_original_missing)
        psnr = peak_signal_noise_ratio(
            x_imputed_missing, x_original_missing, data_range=1.0
        )
        ssim = structural_similarity(
            x_imputed_missing, x_original_missing, data_range=1.0
        )

        results_mse[f"fold{fold}"] = round(mae, 4)  # Stores MAE (Mean Absolute Error)
        results_psnr[f"fold{fold}"] = round(psnr, 4)
        results_ssim[f"fold{fold}"] = round(ssim, 4)

        tf.keras.backend.clear_session()
        del imputer
        gc.collect()

    # Results - MAE, PSNR, SSIM metrics
    results = pd.DataFrame(
        {
            "MAE": results_mse,  # Mean Absolute Error on missing pixels
            "PSNR": results_psnr,  # Peak Signal-to-Noise Ratio
            "SSIM": results_ssim,  # Structural Similarity Index
        }
    )
    results.to_csv(
        f"./results/{model_impt}/{dataset_name}_{model_impt}_{md_mechanism}_results.csv"
    )


if __name__ == "__main__":

    name = "breakhist"
    tempo_total = {}

    # Carregar as imagens (labels e patients na mesma ordem posicional que images)
    data = Datasets(name)
    images, filenames, labels, patients = data.load_data()

    images = np.array(images)
    filenames = np.array(filenames)
    labels = np.array(labels)
    patients = np.array(patients)

    algorithms = ["mat"]
    MD_MECHANISMS = "MCAR"

    for model_impt in algorithms:
        init_time = perf_counter()
        run_experimental_design(
            model_impt, MD_MECHANISMS, images, labels, filenames, patients, name
        )
        end_time = perf_counter()
        tempo_total[f"{model_impt}-{MD_MECHANISMS}"] = round(end_time - init_time, 2)

        res_tempo = pd.DataFrame({"Tempo": tempo_total})
        res_tempo.to_csv(f"./results/tempo_{name}_{model_impt}.csv", index=False)
