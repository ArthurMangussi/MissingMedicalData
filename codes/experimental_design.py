"""
Main code for run Experimental Setup for Missing Data
Imputation into Images (i.e., Image Inpainting)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from codes.data_amputation import ImageDataAmputation
from utils.MyModels import ModelsImputation
from utils.MeLogSingle import MeLogger

_logger = MeLogger()

# Carregar as imagens
(x_train_og, _), (x_test_og, _) = tf.keras.datasets.mnist.load_data()

results_mse = {}
results_psnr = {}
results_ssim = {}

number_of_experiments = 1
for iter in range(number_of_experiments):

    # Holdout simples e pré-processamento das imagens
    x_train_test = np.concatenate((x_train_og, x_test_og), axis=0)
    x_train_val, x_test = train_test_split(x_train_test, test_size=0.2)
    x_train, x_val = train_test_split(x_train_val, test_size=0.2)

    amputation = ImageDataAmputation(missing_rate=0.05)
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
    missing_mask_test_flat = missing_mask_test.astype(bool).flatten()
    mse = mean_squared_error(x_test_imputed.flatten()[missing_mask_test_flat],
                                x_test.flatten()[missing_mask_test_flat])
    # pnsr = peak_signal_noise_ratio()
    # ssim = structural_similarity()
    results_mse[f"iter{iter+1}"] = round(mse, 4)
    # results_psnr[f"iter{iter+1}"] = round(psnr, 4)
    # results_ssim[f"iter{iter+1}"] = round(ssim, 4)
    _logger.info(f"Iteration = {iter+1}")


# Resultados - MSE PSNR 
results = pd.DataFrame({"MSE":results_mse,
                       "PSNR":results_psnr,
                       "SSIM": results_ssim})
results.to_csv("./results/MCAR_results.csv", index=False)