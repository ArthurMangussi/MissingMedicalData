from PIL import Image
import os 
import numpy as np
class Utilities:
    @staticmethod
    def save_image(mechanism:str, missing_rate:str, images:np.ndarray):
        """
        Method to save the array as an image.
        """
        save_dir = f"./results/imputed_images/{mechanism}_{missing_rate}"
        os.makedirs(save_dir)
        for i, img_array in enumerate(images):
            # Assume que img_array está em [0, 1]. Se não, normalize:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

            # Se for 2D (grayscale), converte diretamente
            img = Image.fromarray(img_array)

            img.save(os.path.join(save_dir, f"IMG_{i:04d}.png"))