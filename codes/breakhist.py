"""
Carrega o dataset BreaKHis (imagens histopatológicas de tumores de mama) e
o divide em treino/validação/teste para a tarefa de classificação binária
benigno vs. maligno.

A divisão é feita no nível do paciente (não da imagem): como o BreaKHis tem
vários recortes por lâmina/paciente, dividir por imagem deixaria recortes do
mesmo paciente vazarem entre os conjuntos e inflaria artificialmente o
desempenho.
"""
import sys

sys.path.append("./")

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.MeLogSingle import MeLogger
from utils.MyDataset import CustomImageDataset, Datasets

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # fração do dataset completo reservada para validação


def split_by_patient(
    patients,
    labels,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Constrói índices de treino/validação/teste dividindo por paciente, de
    forma que todas as imagens de um mesmo paciente caiam em um único
    conjunto.
    """
    patients = np.array(patients)
    labels = np.array(labels)

    df_patients = pd.DataFrame({"patient": patients, "label": labels}).drop_duplicates(
        "patient"
    )

    train_val_patients, test_patients = train_test_split(
        df_patients["patient"],
        test_size=test_size,
        random_state=random_state,
        stratify=df_patients["label"],
    )

    train_val_labels = df_patients.set_index("patient").loc[train_val_patients, "label"]
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_labels,
    )

    train_idx = np.where(np.isin(patients, train_patients.values))[0]
    val_idx = np.where(np.isin(patients, val_patients.values))[0]
    test_idx = np.where(np.isin(patients, test_patients.values))[0]

    return train_idx, val_idx, test_idx


def load_and_split_breakhist(magnification: str = None):
    """
    Carrega o BreaKHis e o divide em treino/validação/teste (label binária:
    0 = benigno, 1 = maligno), mantendo todas as imagens de um mesmo
    paciente no mesmo conjunto.
    """
    _logger = MeLogger()

    data = Datasets("breakhist")
    images, _, labels, patients = data._load_breakhist_images(magnification=magnification)
    labels = np.array(labels)

    train_idx, val_idx, test_idx = split_by_patient(patients, labels)

    x_train, y_train = images[train_idx], labels[train_idx]
    x_val, y_val = images[val_idx], labels[val_idx]
    x_test, y_test = images[test_idx], labels[test_idx]

    _logger.info(
        f"[BreaKHis] Treino: {len(x_train)} | Validação: {len(x_val)} | "
        f"Teste: {len(x_test)} imagens"
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_breakhist_datasets(magnification: str = None, transform=None):
    """
    Carrega, divide e empacota o BreaKHis em instâncias de CustomImageDataset
    prontas para uso (classificação benigno vs. maligno).
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_breakhist(
        magnification
    )

    train_dataset = CustomImageDataset(x_train, y_train, transform=transform)
    val_dataset = CustomImageDataset(x_val, y_val, transform=transform)
    test_dataset = CustomImageDataset(x_test, y_test, transform=transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    os.makedirs("./results/breakhist", exist_ok=True)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_breakhist()

    pd.DataFrame(
        {
            "split": ["train"] * len(y_train)
            + ["val"] * len(y_val)
            + ["test"] * len(y_test),
            "label": np.concatenate([y_train, y_val, y_test]),
        }
    ).to_csv("./results/breakhist/split_summary.csv", index=False)

    train_dataset, val_dataset, test_dataset = build_breakhist_datasets()
