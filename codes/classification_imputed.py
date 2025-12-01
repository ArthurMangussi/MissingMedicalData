import sys
sys.path.append("./")
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.MyDataset import CustomImageDataset
import gc
import pandas as pd
import numpy as np 
import tensorflow as tf
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from codes.classification_baseline import NUM_CLASSES, device, BATCH_SIZE, train_model, evaluate_model


def run_pipeline(MODEL_IMPT, MD_MECHANISM, MISSING_RATE):
    _logger = MeLogger()
    _logger.info(f"{MODEL_IMPT} no {MD_MECHANISM} com {MISSING_RATE*100}%")
    
    data = Datasets('inbreast')
    inbreast_images, y = data._load_inbreast_images_imputed(md_mechanism=MD_MECHANISM,
                                                            model_impt=MODEL_IMPT,
                                                            missing_rate=MISSING_RATE)    
    results_accuracy = {}
    results_f1 = {}
    results_roc = {}

    for fold in range(5):
        _logger.info(f"\n[Fold {fold + 1}/5]")
        # Separar dados de teste (1 paciente)
        X_test = inbreast_images[fold]    # shape: (82, 256, 256)
        y_test = y[fold]                  # shape: (82,)

        # Separar dados de treino (outros 4 pacientes)
        X_train_full = np.concatenate([inbreast_images[i] for i in range(5) if i != fold], axis=0)
        y_train_full = np.concatenate([y[i] for i in range(5) if i != fold], axis=0)

        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=fold, stratify=y_train_full
        )
        
        # Treinar a vgg16
        model_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # 1. Encontra o número de features de entrada da última camada (geralmente 4096)
        in_features = model_vgg.classifier[6].in_features

        model_vgg.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

        # Opcional: Descongela as camadas do classificador para treinamento
        for param in model_vgg.classifier.parameters():
            param.requires_grad = True

        # Move o modelo para a GPU, se disponível
        model_vgg = model_vgg.to(device)

        train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.2,
                               saturation=0,
                               hue=0),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
        
        vgg_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),  # Necessário para aplicar transformações de torchvision
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # Normalização padrão do ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Crie os objetos Dataset
        train_dataset = CustomImageDataset(x_train, y_train, transform=train_transform)
        val_dataset = CustomImageDataset(x_val, y_val, transform=vgg_transforms)
        test_dataset = CustomImageDataset(X_test, y_test, transform=vgg_transforms)

        # Crie os DataLoaders para iteração
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
        model_vgg.parameters(), 
        lr=1e-5, 
        weight_decay=1e-4
    )

        train_model(
            model=model_vgg,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=300,
        )
        # Predição
        y_true, y_pred, y_probs = evaluate_model(model_vgg, test_loader, device)

        tf.keras.backend.clear_session()
        del model_vgg
        gc.collect()

        roc_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        _logger.info(f"ROC AUC: {roc_auc:.4f}")

        # 4. Salva métricas
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        _logger.info(f"ROC AUC: {roc_auc:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

        results_accuracy[f"fold{fold}"] = round(acc, 4)
        results_f1[f"fold{fold}"] = round(f1, 4)
        results_roc[f"fold{fold}"] = round(roc_auc, 4)

    results = pd.DataFrame({"ACC": results_accuracy, "F1": results_f1, "AUC_ROC":results_roc})
    results.to_csv(f"./new_results/{MD_MECHANISM}/{MODEL_IMPT}_{MISSING_RATE}_results.csv")

if __name__ == "__main__":
    
    MD_MECHANISM = "MCAR"

    run_pipeline("knn",MD_MECHANISM, 0.05)
    run_pipeline("knn",MD_MECHANISM, 0.10)
    run_pipeline("knn",MD_MECHANISM, 0.20)
    run_pipeline("knn",MD_MECHANISM, 0.30)
    run_pipeline("knn",MD_MECHANISM, 0.40)
    run_pipeline("knn",MD_MECHANISM, 0.50)

    run_pipeline("mc",MD_MECHANISM, 0.05)
    run_pipeline("mc",MD_MECHANISM, 0.10)
    run_pipeline("mc",MD_MECHANISM, 0.20)
    run_pipeline("mc",MD_MECHANISM, 0.30)
    run_pipeline("mc",MD_MECHANISM, 0.40)
    run_pipeline("mc",MD_MECHANISM, 0.50)

    run_pipeline("mice",MD_MECHANISM, 0.05)
    run_pipeline("mice",MD_MECHANISM, 0.10)
    run_pipeline("mice",MD_MECHANISM, 0.20)
    run_pipeline("mice",MD_MECHANISM, 0.30)
    run_pipeline("mice",MD_MECHANISM, 0.40)
    run_pipeline("mice",MD_MECHANISM, 0.50)

    run_pipeline("vaewl",MD_MECHANISM, 0.05)
    run_pipeline("vaewl",MD_MECHANISM, 0.10)
    run_pipeline("vaewl",MD_MECHANISM, 0.20)
    run_pipeline("vaewl",MD_MECHANISM, 0.30)
    run_pipeline("vaewl",MD_MECHANISM, 0.40)
    run_pipeline("vaewl",MD_MECHANISM, 0.50)

    run_pipeline("mae-vit",MD_MECHANISM, 0.05)
    run_pipeline("mae-vit",MD_MECHANISM, 0.10)
    run_pipeline("mae-vit",MD_MECHANISM, 0.20)
    run_pipeline("mae-vit",MD_MECHANISM, 0.30)
    run_pipeline("mae-vit",MD_MECHANISM, 0.40)
    run_pipeline("mae-vit",MD_MECHANISM, 0.50)

    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.05)
    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.10)
    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.20)
    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.30)
    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.40)
    run_pipeline("mae-vit-gan",MD_MECHANISM, 0.50)
    