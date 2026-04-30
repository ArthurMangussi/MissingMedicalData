"""
Main code for run Experimental Setup for Missing Data
Imputation into Images (i.e., Image Inpainting)
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from time import perf_counter
from codes.data_amputation import ImageDataAmputation
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyModels import ModelsImputation
from utils.MyUtils import Utilities


def run_experimental_design(
    model_impt: str,
    md_mechanism: str,
    images: np.ndarray,
    labels: list
):
    os.makedirs(f"./results/{model_impt}", exist_ok=True)
    _logger = MeLogger()
    ut = Utilities()
    results_mse = {}
    results_psnr = {}
    results_ssim = {}

    labels = np.array(labels)  # Convertendo para numpy array para StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(images, labels)):
        _logger.info(f"\n[Fold {fold + 1}/5] - {name}")

        x_train_val, x_test = images[train_val_idx], images[test_idx]
        y_train_val, _ = labels[train_val_idx], labels[test_idx]

        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, _, _ = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=fold
        )

        amputation = ImageDataAmputation()

        x_train, x_train_md, _ = amputation.generate_mcar_dead_pixels(
            x_train, p_single=0.02, p_cluster=0.01, cluster_size=5
        )
        x_val, x_val_md, _ = amputation.generate_mcar_dead_pixels(
            x_val, p_single=0.02, p_cluster=0.01, cluster_size=5
        )
        x_test, x_test_md, missing_mask_test = amputation.generate_mcar_dead_pixels(
            x_test, p_single=0.02, p_cluster=0.01, cluster_size=5
        )

        model = ModelsImputation()
        imputer = model.choose_model(
            model=model_impt,
            x_train=x_train,
            x_val_md=x_val_md,
            x_val=x_val,
            x_train_md=x_train_md,
        )

        if model_impt == "mae-vit" or model_impt == "mae-vit-gan":
            # MAE-ViT expects incomplete image for inpainting
            x_test_imputed = model.mae_imputer_transform(
                model=imputer,
                x_test_md_np=x_test_md,  # Fixed: use x_test_md (incomplete)
                missing_mask_test_np=missing_mask_test,
            )

        elif model_impt == "mc":
            x_test_imputed = imputer.transform(x_test_md, missing_mask_test)

        elif model_impt == "diffusion":
            # Diffusion model - use incomplete image
            prompt = "Full-field digital mammography, high-quality breast parenchyma, no artifacts, no lesions, inpainting task."
            x_test_imputed = model.diffusion_transform(
                model=imputer,
                x_test_md_np=np.squeeze(x_test_md, axis=-1),
                missing_mask_test_np=np.squeeze(missing_mask_test, axis=-1),
                prompt=prompt,
                num_inference_steps=150,
            )
        else:
            # DIP, KNN, MICE, etc.
            if model_impt == "dip":
                batch_size = 16  # Ajuste conforme a memória da sua GPU
                x_test_imputed_all = []

                for i in range(0, len(x_test_md), batch_size):
                    _logger.info(
                        f"DIP: Processando lote {i//batch_size + 1}/{(len(x_test_md) + batch_size - 1) // batch_size}"
                    )

                    # Seleciona o lote atual
                    batch_x = x_test_md[i : i + batch_size]
                    batch_m = missing_mask_test[i : i + batch_size]

                    # Transpose para (N, C, H, W)
                    batch_x_torch = batch_x.transpose(0, 3, 1, 2)
                    batch_m_torch = batch_m.transpose(0, 3, 1, 2)

                    # Executa o DIP no lote todo de uma vez
                    # fit_and_transform agora faz o trabalho pesado
                    imputed_batch = imputer.fit_and_transform(
                        batch_x_torch, batch_m_torch
                    )

                    # Transpose de volta para (N, H, W, C) e guarda
                    x_test_imputed_all.append(imputed_batch.transpose(0, 2, 3, 1))

                # Junta tudo no array final (3000, 224, 224, 1)
                x_test_imputed = np.concatenate(x_test_imputed_all, axis=0)
            else:
                x_test_imputed = imputer.transform(x_test_md)

        ## Save the reconstructed image
        ut.save_image_cbis(
            mechanism=md_mechanism,
            images=x_test_imputed,
            fold=fold,
            model_impt=model_impt,
            dataset=name,
        )

        ## Measure the imputation performance
        # Handle multi-channel images by averaging across channels
        missing_mask_test_binary = missing_mask_test.astype(bool)
        if x_test_imputed.ndim == 4:  # If mask has channel dimension, reduce it
            x_test_imputed = np.squeeze(x_test_imputed, axis=-1)
            missing_mask_test_binary = np.squeeze(missing_mask_test_binary, axis=-1)
        elif x_test_imputed.ndim == 3:
            missing_mask_test_binary = np.squeeze(missing_mask_test_binary, axis=-1)

        # Extract only missing pixels from both images
        x_imputed_missing = x_test_imputed[missing_mask_test_binary]
        x_original_missing = x_test[missing_mask_test_binary]

        # Compute MAE on missing pixels only
        mae = mean_absolute_error(x_imputed_missing, x_original_missing)

        # For PSNR/SSIM, compute per channel if multi-channel, then average
        if len(x_test.shape) > 2 and x_test.shape[0] > 1:
            # Multi-channel: compute per-channel metrics and average
            psnr_values = []
            ssim_values = []
            for c in range(x_test.shape[0]):
                num_pixels_missing = missing_mask_test_binary[c].sum()

                if num_pixels_missing == 0:
                    print(
                        f"Pulo na imagem {c}: O artefato caiu fora do tecido mamário."
                    )
                    continue

                psnr_c = peak_signal_noise_ratio(
                    x_test_imputed[c, missing_mask_test_binary[c]],
                    x_test[c, missing_mask_test_binary[c]].reshape(-1),
                    data_range=1.0,
                )
                ssim_c = structural_similarity(
                    x_test_imputed[c, missing_mask_test_binary[c]],
                    x_test[c, missing_mask_test_binary[c]].reshape(-1),
                    data_range=1.0,
                )
                psnr_values.append(psnr_c)
                ssim_values.append(ssim_c)
            psnr = np.mean(psnr_values)
            ssim = np.mean(ssim_values)
        else:
            # Single channel or 2D image
            psnr = peak_signal_noise_ratio(
                x_imputed_missing, x_original_missing, data_range=1.0
            )
            ssim = structural_similarity(
                x_test_imputed[missing_mask_test_binary],
                x_test[missing_mask_test_binary],
                data_range=1.0,
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
            "SSIM": results_ssim,
        }
    )  # Structural Similarity Index
    results.to_csv(
        f"./results/{model_impt}/{name}_{model_impt}_{md_mechanism}_results.csv"
    )


if __name__ == "__main__":

    dataset_names = ["cbis-ddsm"]  #
    tempo_total = {}
    for name in dataset_names:
        # Carregar as imagens
        data = Datasets(name)
        inbreast_images, _, labels = data.load_data()

        algorithms = ["vaewl"]
        MD_MECHANISMS = "MCAR"

        for model_impt in algorithms:
            init_time = perf_counter()
            run_experimental_design(
                model_impt, MD_MECHANISMS, inbreast_images, labels
            )
            end_time = perf_counter()
            tempo_total[f"{model_impt}-{MD_MECHANISMS}"] = round(
                end_time - init_time, 2
            )

            res_tempo = pd.DataFrame({"Tempo": tempo_total})
            res_tempo.to_csv(f"./results/tempo_{name}_{model_impt}.csv", index=False)
