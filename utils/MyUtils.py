from PIL import Image
import os 
import numpy as np
import pandas as pd

class Utilities:
    @staticmethod
    def save_image(mechanism:str, 
                   missing_rate:str, 
                   images:np.ndarray, 
                   fold:int, 
                   model_impt:str,
                   labels_names:dict,
                   image_ids:list):
        """
        Method to save the array as an image.
        """
        save_dir = f"./results/{model_impt}/imputed_images/fold{fold}_{mechanism}_{missing_rate}"
        os.makedirs(save_dir, exist_ok=True)
        
        labels = np.array([{str(i):labels_names[i]} for i in image_ids])
        
        for count, image in enumerate(images):
            img = np.squeeze(image)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Se for 2D (grayscale), converte diretamente
            img_pil = Image.fromarray(img)
            if img_pil.mode != 'L':
                img_pil = img_pil.convert('L')

            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))
        
        labels_df = pd.DataFrame(
            [(k, v) for d in labels for k, v in d.items()],
            columns=['Image', 'Target']
        )
        labels_df.to_csv(os.path.join(save_dir, "classes.csv"))