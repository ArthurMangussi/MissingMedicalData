import sys
sys.path.append("./")
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyUtils import Utilities
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from utils.MyModels import CNN
import pandas as pd
import numpy as np 

# Baseline Original Images
data = Datasets('inbreast')
inbreast_images, y = data.load_data()

_logger = MeLogger()
ut = Utilities()
results_accuracy = {}
results_f1 = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_val_idx, test_idx) in enumerate(skf.split(inbreast_images, y)):
    _logger.info(f"\n[Fold {fold + 1}/5]")

    x_train_val, x_test = inbreast_images[train_val_idx], inbreast_images[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]

    # Divide treino e validação internamente (ex: 20% para validação)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=fold
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
    y_pred = model.predict(x_test=x_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    print("Threshold:", best_threshold)

    y_pred = 1*(y_pred > best_threshold)

    # Salva métricas
    acc = accuracy_score(y_true=y_test,
                        y_pred=y_pred)
    f1 = f1_score(y_true=y_test,
                  y_pred=y_pred)
    results_accuracy[f"fold{fold}"] = round(acc,4)
    results_f1[f"fold{fold}"] = round(f1,4)

results = pd.DataFrame({"ACC":results_accuracy,
                        "F1": results_f1})

results.to_csv(f"./results/baseline_results.csv")