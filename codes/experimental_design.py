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

from codes.data_amputation import ImageDataAmputation
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyModels import ModelsImputation
from utils.MyUtils import Utilities

def run_experimental_design(model_impt:str,
                            missing_rate: float,
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

        amputation = ImageDataAmputation(missing_rate=missing_rate)
        
        if md_mechanism == "MCAR":
            x_train, x_train_md, missing_mask_train = amputation.generate_missing_mask_mcar(x_train)
            x_val, x_val_md, _ = amputation.generate_missing_mask_mcar(x_val)
            x_test, x_test_md, missing_mask_test = amputation.generate_missing_mask_mcar(x_test)
        elif md_mechanism == "MNAR":
            x_train, x_train_md, missing_mask_train = amputation.generate_missing_mask_mnar(x_train)
            x_val, x_val_md, _ = amputation.generate_missing_mask_mnar(x_val)
            x_test, x_test_md, missing_mask_test = amputation.generate_missing_mask_mnar(x_test)
        else:
            x_test, x_test_md, missing_mask_test = amputation.generate_random_squares_mask(x_test)


        model = ModelsImputation()
        imputer = model.choose_model(model=model_impt, 
                                    x_train=x_train,
                                    x_train_md=x_train_md,
                                    x_val_md=x_val_md,
                                    x_val=x_val,
                                    mask_train=missing_mask_train
                                    )

        if model_impt == "mae-vit" or model_impt == "mae-vit-gan":

            x_test_imputed = model.mae_imputer_transform(model=imputer,
                                        x_test_md_np=x_test,
                                        missing_mask_test_np=missing_mask_test,
                                        missing_rate=missing_rate)
        
        elif model_impt == "mc":
            x_test_imputed = imputer.transform(x_test_md, missing_mask_test)
        else:
            x_test_imputed = imputer.transform(x_test_md)
          
        ## Save the reconstructed image
        ut.save_image(mechanism=md_mechanism,
                    missing_rate=missing_rate,
                    images=x_test_imputed,
                    fold=fold,
                    model_impt=model_impt,
                    labels_names= labels_names, 
                    image_ids = img_test_idx,
                    dataset=name)

        ## Measure the imputation performance
        missing_mask_test_flat = missing_mask_test.astype(bool).flatten()

        mse = mean_absolute_error(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat])
        psnr = peak_signal_noise_ratio(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat],
                                    data_range=1.0)
        ssim = structural_similarity(x_test_imputed.flatten()[missing_mask_test_flat],
                                    x_test.flatten()[missing_mask_test_flat],
                                    data_range=1.0)
        
        results_mse[f"fold{fold}"] = round(mse, 4)
        results_psnr[f"fold{fold}"] = round(psnr, 4)
        results_ssim[f"fold{fold}"] = round(ssim, 4)

        tf.keras.backend.clear_session()
        del imputer
        gc.collect()


    # Resultados - MSE PSNR 
    results = pd.DataFrame({"MSE":results_mse,
                        "PSNR":results_psnr,
                        "SSIM": results_ssim})
    results.to_csv(f"./results/{model_impt}/{name}_{model_impt}_{md_mechanism}_{missing_rate}_results.csv")

if __name__ == "__main__":
    
    dataset_names = ["vindr-reduzido", "mias", "inbreast"] 

    for name in dataset_names:
        # Carregar as imagens
        data = Datasets(name)
        inbreast_images, y_mapped, image_ids = data.load_data()
        
        
        algorithms = ["mc","knn", "mice", "mae-vit", "mae-vit-gan"]
        MD_MECHANISM = ["MCAR", "MNAR"]

        for md_mechanism in MD_MECHANISM:
            for model_impt in algorithms:
                run_experimental_design(model_impt,0.05,md_mechanism,inbreast_images, y_mapped, image_ids)
                run_experimental_design(model_impt,0.10,md_mechanism,inbreast_images, y_mapped, image_ids)
                run_experimental_design(model_impt,0.20,md_mechanism,inbreast_images, y_mapped, image_ids)
                run_experimental_design(model_impt,0.30,md_mechanism,inbreast_images, y_mapped, image_ids)
                run_experimental_design(model_impt,0.40,md_mechanism,inbreast_images, y_mapped, image_ids)
                run_experimental_design(model_impt,0.50,md_mechanism,inbreast_images, y_mapped, image_ids)
