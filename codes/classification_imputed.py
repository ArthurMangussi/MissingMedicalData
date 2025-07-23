import sys
sys.path.append("./")
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyUtils import Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from utils.MyModels import CNN
import gc
import pandas as pd
import numpy as np 
import tensorflow as tf
import multiprocessing

def run_pipeline(MODEL_IMPT, MD_MECHANISM, MISSING_RATE):
    _logger = MeLogger()
    _logger.info(f"{MODEL_IMPT} no {MD_MECHANISM} com {MISSING_RATE*100}%")
    # Baseline Original Images
    data = Datasets('inbreast')
    inbreast_images, y = data._load_inbreast_images_imputed(md_mechanism=MD_MECHANISM,
                                                            model_impt=MODEL_IMPT,
                                                            missing_rate=MISSING_RATE)
    _logger = MeLogger()
    ut = Utilities()
    results_accuracy = {}
    results_f1 = {}

    for fold in range(5):
        _logger.info(f"\n[Fold {fold + 1}/5]")
        # Separar dados de teste (1 paciente)
        X_test = inbreast_images[fold]    # shape: (82, 256, 256)
        y_test = y[fold]                  # shape: (82,)

        # Separar dados de treino (outros 4 pacientes)
        X_train = np.concatenate([inbreast_images[i] for i in range(5) if i != fold], axis=0)
        y_train = np.concatenate([y[i] for i in range(5) if i != fold], axis=0)

        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=fold
    )
        
        # Treinar a CNN
        model = CNN(img_shape=(256,256),
                    batch_size=32,
                    epochs=500)
        model.fit(x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val)


        # Predição
        y_pred = model.predict(x_test=X_test)

        if fold == 0:
            best_threshold = 0.27
        elif fold == 1:
            best_threshold = 0.372
        elif fold == 2:
            best_threshold = 0.318
        elif fold == 3:
            best_threshold = 0.346
        elif fold == 4:
            best_threshold = 0.381
        
        y_pred = 1*(y_pred > best_threshold)

        # Salva métricas
        acc = accuracy_score(y_true=y_test,
                            y_pred=y_pred)
        f1 = f1_score(y_true=y_test,
                    y_pred=y_pred)
        results_accuracy[f"fold{fold}"] = round(acc,4)
        results_f1[f"fold{fold}"] = round(f1,4)

        tf.keras.backend.clear_session()
        del model
        gc.collect()

    results = pd.DataFrame({"ACC":results_accuracy,
                            "F1": results_f1})

    results.to_csv(f"./results/{MODEL_IMPT}_{MD_MECHANISM}_{MISSING_RATE}_results.csv")

if __name__ == "__main__":

    for MD_MECHANISM in ["MCAR", "MNAR"]:
        for model_impt in ["knn", "mc", "vaewl"]:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

                args_list = [
                            
                            (model_impt,MD_MECHANISM, 0.05),
                            (model_impt,MD_MECHANISM, 0.10),
                            (model_impt,MD_MECHANISM, 0.20),
                            ]
                
                pool.starmap(run_pipeline,args_list)