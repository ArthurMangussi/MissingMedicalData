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

def run_experimental_design(model_impt:str,
                            md_mechanism: str,
                            images: np.ndarray,
                            labels_names:dict, 
                            image_ids:list):
    os.makedirs(f"./results/{model_impt}", exist_ok=True)
    _logger = MeLogger()
    ut = Utilities()
    results_mse = {}
    results_psnr = {}
    results_ssim = {}


    image_ids = np.array(image_ids)

    if name== "inbreast":
        
        labels = np.array([labels_names[i] for i in image_ids]) 
    else:
        labels = np.array(list(labels_names.values()))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(images, labels)):
        _logger.info(f"\n[Fold {fold + 1}/5]")

        x_train_val, x_test = images[train_val_idx], images[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        img_test_idx = image_ids[test_idx]
        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=fold
    )

        amputation = ImageDataAmputation()
        
        x_train, x_train_md, _ = amputation.generate_mar_truncation(x_train)
        x_val, x_val_md, _ = amputation.generate_mar_truncation(x_val)
        x_test, x_test_md, missing_mask_test = amputation.generate_mar_truncation(x_test )


        model = ModelsImputation()
        imputer = model.choose_model(model=model_impt, 
                                    x_train=x_train,
                                    x_val_md=x_val_md,
                                    x_val=x_val,
                                    x_train_md=x_train_md,
                                    )

        if model_impt == "mae-vit" or model_impt == "mae-vit-gan":
            # MAE-ViT expects incomplete image for inpainting
            x_test_imputed = model.mae_imputer_transform(model=imputer,
                                        x_test_md_np=x_test_md,  # Fixed: use x_test_md (incomplete)
                                        missing_mask_test_np=missing_mask_test,
                                        missing_rate=missing_rate)

        elif model_impt == "mc":
            x_test_imputed = imputer.transform(x_test_md, missing_mask_test)

        elif model_impt == "diffusion":
            # Diffusion model - use incomplete image
            x_test_imputed = model.diffusion_transform(model=imputer,
                                                       x_test_md_np=x_test_md,
                                                       missing_mask_test_np=missing_mask_test,
                                                       prompt="medical image",
                                                       num_inference_steps=20)
        else:
            # DIP, KNN, MICE, etc. - use incomplete image
            x_test_imputed = imputer.transform(x_test_md)
          
        ## Save the reconstructed image
        ut.save_image(mechanism=md_mechanism,
                    images=x_test_imputed,
                    fold=fold,
                    model_impt=model_impt,
                    labels_names= labels_names, 
                    image_ids = img_test_idx,
                    dataset=name)

        ## Measure the imputation performance
        # Handle multi-channel images by averaging across channels
        missing_mask_test_binary = missing_mask_test.astype(bool)

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
                psnr_c = peak_signal_noise_ratio(
                    x_test_imputed[c, missing_mask_test_binary[c]],
                    x_test[c, missing_mask_test_binary[c]],
                    data_range=1.0
                )
                ssim_c = structural_similarity(
                    x_test_imputed[c][missing_mask_test_binary[c].reshape(x_test[c].shape)],
                    x_test[c][missing_mask_test_binary[c].reshape(x_test[c].shape)],
                    data_range=1.0
                )
                psnr_values.append(psnr_c)
                ssim_values.append(ssim_c)
            psnr = np.mean(psnr_values)
            ssim = np.mean(ssim_values)
        else:
            # Single channel or 2D image
            psnr = peak_signal_noise_ratio(
                x_imputed_missing,
                x_original_missing,
                data_range=1.0
            )
            ssim = structural_similarity(
                x_test_imputed[missing_mask_test_binary],
                x_test[missing_mask_test_binary],
                data_range=1.0
            )
        
        results_mse[f"fold{fold}"] = round(mae, 4)  # Stores MAE (Mean Absolute Error)
        results_psnr[f"fold{fold}"] = round(psnr, 4)
        results_ssim[f"fold{fold}"] = round(ssim, 4)

        tf.keras.backend.clear_session()
        del imputer
        gc.collect()


    # Results - MAE, PSNR, SSIM metrics
    results = pd.DataFrame({"MAE":results_mse,  # Mean Absolute Error on missing pixels
                        "PSNR":results_psnr,    # Peak Signal-to-Noise Ratio
                        "SSIM": results_ssim})  # Structural Similarity Index
    results.to_csv(f"./results/{model_impt}/{name}_{model_impt}_{md_mechanism}_{missing_rate}_results.csv")

if __name__ == "__main__":
    
    dataset_names = ["inbreast"] 
    tempo_total = {}
    for name in dataset_names:
        # Carregar as imagens
        data = Datasets(name)
        inbreast_images, y_mapped, image_ids = data.load_data()
        
        
        algorithms = ["knn"]
        MD_MECHANISMS = "MAR-Truncation"
        
        for model_impt in algorithms:
                init_time = perf_counter()
                run_experimental_design(model_impt,0.05,MD_MECHANISMS,inbreast_images, y_mapped, image_ids)
                end_time = perf_counter()
                tempo_total[f"{model_impt}-{MD_MECHANISMS}"] = round(end_time-init_time, 2)
                
    res_tempo = pd.DataFrame({"Tempo": tempo_total})
    res_tempo.to_csv("tempo.csv", index=False)
                
