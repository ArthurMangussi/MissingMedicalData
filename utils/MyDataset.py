import os

import cv2
import numpy as np
import pandas as pd
import pydicom

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Datasets:
    def __init__(self, name_dataset: str):
        self.name_dataset = name_dataset

    @staticmethod
    def _load_inbreast_images():
        """
        Method to load the INBreast dataset.

        Args:
            size (tuple): The shape for the redimension
        image
        """
        images = []
        filenames = []

        data_dir = "/home/gpu-10-2025/Área de trabalho/Datasets/INBreast/PNG/"
        # Ordenar os arquivos para manter consistência
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

        for f in files:
            caminho_imagem = os.path.join(data_dir, f)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                raise ValueError(f"Erro ao ler a imagem: {f}")

            # Redimensiona
            imagem = cv2.resize(imagem, (224, 224))
            images.append(imagem)
            complete_name = f.split("_")
            filenames.append(
                complete_name[0]
                + "_"
                + complete_name[3]
                + "_"
                + complete_name[4]
                + "_"
                + complete_name[5][:-4]
            )  # mantém o nome completo como string

        return np.array(images), [i.replace("ML", "MLO") for i in filenames]

    @staticmethod
    def _load_inbreast_images_imputed(
        md_mechanism: str, model_impt: str, missing_rate: float
    ):
        """
        Method to load the INBreast dataset after the imputation
        process.

        """
        data_dir = f"/home/mult-e/Área de trabalho/MissingMedicalData/results/{model_impt}/imputed_images/"

        fold0, fold1, fold2, fold3, fold4, labels = [], [], [], [], [], []
        for fold in range(5):
            impt_dir = data_dir + f"fold{fold}_{md_mechanism}_{missing_rate}"
            # Percorre todos os arquivos da pasta
            for nome_arquivo in os.listdir(impt_dir):
                if nome_arquivo.endswith(".png"):
                    caminho_imagem = os.path.join(impt_dir, nome_arquivo)

                    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

                    if fold == 0:
                        fold0.append(imagem)

                    elif fold == 1:
                        fold1.append(imagem)

                    elif fold == 2:
                        fold2.append(imagem)

                    elif fold == 3:
                        fold3.append(imagem)

                    elif fold == 4:
                        fold4.append(imagem)

                elif nome_arquivo.endswith(".csv"):
                    caminho_csv = impt_dir + "/classes.csv"
                    df = pd.read_csv(caminho_csv)
                    labels.append(df.iloc[:, 1].values)

                else:
                    raise ValueError(f"Erro ao ler a imagem: {nome_arquivo}")

        images = [fold0, fold1, fold2, fold3, fold4]
        return np.array(images), np.array(labels)

    @staticmethod
    def _load_inbreast_labels():
        label_file = (
            "/home/gpu-10-2025/Área de trabalho/Datasets/INBreast/INbreast.xlsx"
        )
        df = pd.read_excel(label_file)  # arquivo com mapping de nome -> label
        # Garantindo que todos os valores sejam strings
        keys = (
            df["File Name"].astype(str)
            + "_"
            + df["Laterality"].astype(str)
            + "_"
            + df["View"].astype(str)
            + "_"
            + "ANON"
        )

        # Criando o dicionário
        my_dict = dict(zip(keys, df["Target"]))

        return my_dict

    def load_data(self):
        match self.name_dataset:
            case "inbreast":
                images, image_ids = self._load_inbreast_images()
                y_dict = self._load_inbreast_labels()

                return images, y_dict, image_ids


class CustomImageDataset(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        # Converte os arrays NumPy para tensores PyTorch
        self.images = torch.tensor(images_array, dtype=torch.float32).unsqueeze(
            1
        )  # Assumindo INBREAST é Grayscale (1 canal)
        self.labels = torch.tensor(labels_array, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Se a VGG16 espera 3 canais, você precisa converter de 1 canal para 3 aqui
        if (
            image.shape[0] == 1
            and self.transform is not None
            and len(self.transform.transforms) > 0
            and "Normalize" in str(self.transform.transforms[-1])
        ):
            # Repete o canal grayscale 3 vezes: (1, H, W) -> (3, H, W)
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(
                image
            )  # Aplica todas as transformações (Resize, Crop, Normalizar)

        return image, label
