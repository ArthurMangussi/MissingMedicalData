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
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F  # Necessário para Softmax/Sigmoid


# --- CONFIGURAÇÕES ---
NUM_CLASSES = 2
BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.counter = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)

# --- FUNÇÃO DE TREINAMENTO AJUSTADA ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Ajustada para receber o DataLoader diretamente."""
    early_stopping = EarlyStopping(patience=30, min_delta=1e-4)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Treinamento
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

        # Validação
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early Stopping Check
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\n Early stopping at epoch {epoch+1}")
            break

    return model


# --- NOVA FUNÇÃO DE AVALIAÇÃO (CRÍTICA) ---
def evaluate_model(model, test_loader, device, num_classes=2):
    """
    Avalia o modelo no conjunto de teste e retorna:
        - y_true: labels verdadeiros
        - y_pred: classes previstas
        - y_probs: probabilidades da classe positiva (binário)
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim = 1)

            _, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            if num_classes == 2:
                y_probs.extend(probs[:, 1].cpu().numpy())
            else:
                y_probs.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)

if __name__ == "__main__":

    # Baseline Original Images
    data = Datasets("inbreast")
    inbreast_images, labels_names, img_ids = data.load_data()

    image_ids = np.array(img_ids)
    labels = np.array([labels_names[i] for i in image_ids])

    _logger = MeLogger()
    ut = Utilities()
    results_accuracy = {}
    results_f1 = {}
    results_roc = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    for fold, (train_val_idx, test_idx) in enumerate(skf.split(inbreast_images, labels)):
        _logger.info(f"\n[Fold {fold + 1}/5]")

        x_train_val, x_test = inbreast_images[train_val_idx], inbreast_images[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        # Divide treino e validação internamente (ex: 20% para validação)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=fold, stratify=y_train_val
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
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                # Normalização padrão do ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Crie os objetos Dataset
        train_dataset = CustomImageDataset(x_train, y_train, transform=train_transform)
        val_dataset = CustomImageDataset(x_val, y_val, transform=vgg_transforms)
        test_dataset = CustomImageDataset(x_test, y_test, transform=vgg_transforms)

        class_sample_count = np.array([sum(y_train == t) for t in np.unique(y_train)])

        # Peso inversamente proporcional à frequência da classe
        weight = 1. / class_sample_count

        # Cria vetor de pesos por amostra
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight).double()

        sampler = WeightedRandomSampler(weights=samples_weight,
                                        num_samples=len(samples_weight),
                                        replacement=True)

        # Crie os DataLoaders para iteração
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
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

        roc_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        _logger.info(f"ROC AUC: {roc_auc:.4f}")

        # 4. Salva métricas
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        results_accuracy[f"fold{fold}"] = round(acc, 4)
        results_f1[f"fold{fold}"] = round(f1, 4)
        results_roc[f"fold{fold}"] = round(roc_auc, 4)

    results = pd.DataFrame({"ACC": results_accuracy, "F1": results_f1, "AUC_ROC":results_roc})

    results.to_csv(f"./results/baseline_results.csv")
