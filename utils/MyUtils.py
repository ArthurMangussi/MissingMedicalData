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
        os.makedirs(save_dir, exist_ok=True)
        for count, image in enumerate(images):
            img = np.squeeze(image)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Se for 2D (grayscale), converte diretamente
            img_pil = Image.fromarray(img)

            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))