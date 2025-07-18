import sys
sys.path.append("./")
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyUtils import Utilities
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from utils.MyModels import CNN

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
                batch_size=64,
                epochs=400)
    model.fit(x_train=x_train,
              y_train=y_train,
              x_val=x_val,
              y_val=y_val)


    # Predição
    y_pred = model.predict(x_test=x_test)

    # Salva métricas