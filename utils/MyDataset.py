import os

import cv2
import numpy as np
import pandas as pd
import pydicom


class Datasets:
    def __init__(self, name_dataset:str):
        self.name_dataset = name_dataset
    
    @staticmethod
    def _load_inbreast_images(size=(256,256)):
        """
        Method to load the INBreast dataset.

        Args:
            size (tuple): The shape for the redimension
        image
        """
        data_dir = "D:\@MamoImages\INBreast\AllDICOMs"
        images = []
        filenames = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]
        edited_filenames = [int(f.split("_")[0]) for f in filenames]

        for file in filenames:
            dicom_path = os.path.join(data_dir, file)
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)
            # Normalização simples
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            images.append(img)
        
        return np.array(images), edited_filenames
    
    @staticmethod
    def _load_inbreast_labels(filenames):
        label_file = "D:\@MamoImages\INBreast\INbreast.xlsx"
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
    
    