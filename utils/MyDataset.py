import os

import cv2
import numpy as np
import pandas as pd
import pydicom


class Datasets:
    def __init__(self, name_dataset:str):
        self.name_dataset = name_dataset
    
    @staticmethod
    def _load_inbreast_images():
        """
        Method to load the INBreast dataset.

        Args:
            size (tuple): The shape for the redimension
        image
        """
        data_dir = "/home/mult-e/Área de trabalho/@MamoImages/INBreast/AllPNG/"
        images = []
        filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        edited_filenames = [int(f.split("_")[0]) for f in filenames]

        # Percorre todos os arquivos da pasta
        for nome_arquivo in os.listdir(data_dir):
            if nome_arquivo.endswith('.png'):
                caminho_imagem = os.path.join(data_dir, nome_arquivo)
                
                imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

                if imagem is not None:
                    images.append(imagem)
                else:
                    raise ValueError(f"Erro ao ler a imagem: {nome_arquivo}")
                
        return np.array(images), edited_filenames
    
    @staticmethod
    def _load_inbreast_labels(filenames):
        label_file = "/home/mult-e/Área de trabalho/@MamoImages/INBreast/INbreast.xlsx"
        df = pd.read_excel(label_file)  # arquivo com mapping de nome -> label
        label_dict = dict(zip(df["File Name"], df["Bi-Rads"]))  # adapte os nomes das colunas
        
        labels = [label_dict.get(name, -1) for name in filenames]
        return np.array(labels)

    def load_data(self):
        match self.name_dataset:
            case "inbreast":
                images, filenames = self._load_inbreast_images()
                y = self._load_inbreast_labels(filenames)
                return images, y    
    
    