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
    def _load_inbreast_images_imputed(md_mechanism:str,
                                      model_impt:str,
                                      missing_rate:float):
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
                if nome_arquivo.endswith('.png'):
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
                    labels.append(df.iloc[:,1].values)

                else:
                    raise ValueError(f"Erro ao ler a imagem: {nome_arquivo}")
                
        images = [fold0, fold1, fold2, fold3, fold4]
        return np.array(images), np.array(labels)
    
    @staticmethod
    def _load_inbreast_labels(filenames):
        label_file = "/home/mult-e/Área de trabalho/@MamoImages/INBreast/INbreast.xlsx"
        df = pd.read_excel(label_file)  # arquivo com mapping de nome -> label
        label_dict = dict(zip(df["File Name"], df["Target"]))  # adapte os nomes das colunas
        
        labels = [label_dict.get(name, -1) for name in filenames]
        return np.array(labels)

    def load_data(self):
        match self.name_dataset:
            case "inbreast":
                images, filenames = self._load_inbreast_images()
                y = self._load_inbreast_labels(filenames)
                return images, y    
    
    