"""
Main code for run Experimental Setup for Missing Data
Imputation into Images (i.e., Image Inpainting)
"""
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from codes.data_amputation import ImageDataAmputation
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyModels import ModelsImputation
from utils.MyUtils import Utilities


def run_experimental_design(missing_rate: float,
                            md_mechanism: str,
                            images: np.ndarray,
                            labels: np.ndarray):
    _logger = MeLogger()
    ut = Utilities()
    results_mse = {}
    results_psnr = {}
    results_ssim = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(images, labels)):
        _logger.info(f"\n[Fold {fold + 1}/5]")

        x_train_val, x_test = images[train_val_idx], images[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=fold
    )

        amputation = ImageDataAmputation(missing_rate=missing_rate)
        x_train, x_train_md, _ = amputation.generate_missing_mask_mcar(x_train)
        x_val, x_val_md, _ = amputation.generate_missing_mask_mcar(x_val)
        x_test, x_test_md, missing_mask_test = amputation.generate_missing_mask_mcar(x_test)

        model = ModelsImputation()
        imputer = model.choose_model("vaewl", 
                                    x_train=x_train,
                                    x_train_md=x_train_md,
                                    x_val_md=x_val_md,
                                    x_val=x_val
                                    )

        x_test_imputed = imputer.transform(x_test_md)

        ## Save the reconstructed image
        ut.save_image(mechanism=md_mechanism,
                    missing_rate=missing_rate,
                    images=x_test_imputed)

        ## Measure the imputation performance
        missing_mask_test_flat = missing_mask_test.astype(bool).flatten()

        mse = mean_squared_error(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat])
        psnr = peak_signal_noise_ratio(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat])
        ssim = structural_similarity(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat],
                                    data_range=1.0)
        
        results_mse[f"iter{iter+1}"] = round(mse, 4)
        results_psnr[f"iter{iter+1}"] = round(psnr, 4)
        results_ssim[f"iter{iter+1}"] = round(ssim, 4)
        
        _logger.info(f"Iteration = {iter+1}")

        tf.keras.backend.clear_session()
        del imputer
        gc.collect()


    # Resultados - MSE PSNR 
    results = pd.DataFrame({"MSE":results_mse,
                        "PSNR":results_psnr,
                        "SSIM": results_ssim})
    results.to_csv(f"./results/MCAR_{missing_rate}_results.csv", index=False)

if __name__ == "__main__":
    MD_MECHANISM = "MCAR"

    # Carregar as imagens
    data = Datasets('inbreast')
    inbreast_images, y = data.load_data()
    
    run_experimental_design(missing_rate=0.05,
                            md_mechanism=MD_MECHANISM,
                            images=inbreast_images,
                            labels=y)
    run_experimental_design(missing_rate=0.10,
                            md_mechanism=MD_MECHANISM,
                            images=inbreast_images,
                            labels=y)
    run_experimental_design(missing_rate=0.20,
                            md_mechanism=MD_MECHANISM,
                            images=inbreast_images,
                            labels=y)