import sys

sys.path.append("./")
from utils.MeLogSingle import MeLogger
from utils.MyDataset import Datasets
from utils.MyUtils import Utilities
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from utils.MyDataset import CustomImageDataset

import pandas as pd
import numpy as np

from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # Necessário para Softmax/Sigmoid


# --- CONFIGURAÇÕES ---
NUM_CLASSES = 2
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- FUNÇÃO DE TREINAMENTO AJUSTADA ---
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Ajustada para receber o DataLoader diretamente."""
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Treinamento
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


# --- NOVA FUNÇÃO DE AVALIAÇÃO (CRÍTICA) ---
def evaluate_model(model, test_loader, device):
    """Coleta todas as predições e labels para avaliação de métricas."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # 1. Forward pass
            outputs = model(inputs).cpu()

            # 2. Coleta de probabilidades (necessário para ROC/AUC)
            # Para 2 classes (Binário), usamos Sigmoid ou Softmax e pegamos a prob da classe positiva (índice 1)
            if NUM_CLASSES == 2:
                probs = F.softmax(outputs, dim=1)[:, 1]
            else:  # Para mais de 2 classes, você precisaria de um tratamento diferente para AUC multi-classe
                probs = F.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


# Baseline Original Images
data = Datasets("inbreast")
inbreast_images, labels_names, img_ids = data.load_data()

image_ids = np.array(img_ids)
labels = np.array([labels_names[i] for i in image_ids])

_logger = MeLogger()
ut = Utilities()
results_accuracy = {}
results_f1 = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

for fold, (train_val_idx, test_idx) in enumerate(skf.split(inbreast_images, labels)):
    _logger.info(f"\n[Fold {fold + 1}/5]")

    x_train_val, x_test = inbreast_images[train_val_idx], inbreast_images[test_idx]
    y_train_val, y_test = labels[train_val_idx], labels[test_idx]

    # Divide treino e validação internamente (ex: 20% para validação)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=fold
    )

    # Treinar a vgg16
    
    # 2. Opcional, mas recomendado: Congela os parâmetros do extrator de features
    #    (as camadas convolucionais) para que apenas as camadas de classificação sejam treinadas
    for param in model_vgg.parameters():
        param.requires_grad = False

    # 1. Encontra o número de features de entrada da última camada (geralmente 4096)
    in_features = model_vgg.classifier[6].in_features

    # 2. Substitui a camada final (índice 6) por uma nova camada linear
    #    que mapeia as features (4096) para o seu número de classes (e.g., 5)
    model_vgg.classifier[6] = nn.Linear(in_features, NUM_CLASSES)

    # Opcional: Descongela as camadas do classificador para treinamento
    for param in model_vgg.classifier.parameters():
        param.requires_grad = True

    # Move o modelo para a GPU, se disponível
    model_vgg = model_vgg.to(device)

    vgg_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),  # Necessário para aplicar transformações de torchvision
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalização padrão do ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Crie os objetos Dataset
    train_dataset = CustomImageDataset(x_train, y_train, transform=vgg_transforms)
    val_dataset = CustomImageDataset(x_val, y_val, transform=vgg_transforms)
    test_dataset = CustomImageDataset(x_test, y_test, transform=vgg_transforms)

    # Crie os DataLoaders para iteração
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Seu VGG16 agora treinará usando o 'train_loader'
    criterion = nn.CrossEntropyLoss()

    # Escolhe apenas os parâmetros que estão DESCONGELADOS (se você congelou o 'features')
    # Caso contrário, use model_vgg.parameters() para treinar tudo.
    optimizer = optim.SGD(model_vgg.classifier.parameters(), lr=0.001, momentum=0.9)

    train_model(
        model=model_vgg,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
    )
    # Predição
    y_pred, y_probs = evaluate_model(model_vgg, test_loader, device)

    roc_auc = roc_auc_score(y_true=y_pred, y_score=y_probs)
    _logger.info(f"ROC AUC: {roc_auc:.4f}")

    # 4. Salva métricas
    acc = accuracy_score(y_true=y_pred, y_pred=y_test)
    f1 = f1_score(y_true=y_pred, y_pred=y_test)

    results_accuracy[f"fold{fold}"] = round(acc, 4)
    results_f1[f"fold{fold}"] = round(f1, 4)

results = pd.DataFrame({"ACC": results_accuracy, "F1": results_f1})

results.to_csv(f"./results/baseline_results.csv")
