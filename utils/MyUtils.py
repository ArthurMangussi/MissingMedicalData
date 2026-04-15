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
                   image_ids:list,
                   dataset:str):
        """
        Method to save the array as an image.
        """
        save_dir = f"./new_results/{dataset}/{model_impt}/imputed_images/fold{fold}_{mechanism}_{missing_rate}"
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            labels = np.array([{str(i):labels_names[i]} for i in image_ids])
        except KeyError:
            image_ids = [i[:-4]for i in image_ids]
            labels = np.array([{str(i):labels_names[i]} for i in image_ids])

        
        for count, image in enumerate(images):
            img = np.array(image, copy=True)

            # Remove accidental batch dimension: (1, H, W, C) or (1, C, H, W)
            if img.ndim == 4:
                img = img[0]

            # Normalize to [0, 255] uint8 before reshaping
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            if img.ndim == 3:
                # Detect channel-first layout (C, H, W) when C is small (1 or 3)
                # and clearly smaller than spatial dimensions
                if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                    img = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

                # Collapse single channel: (H, W, 1) → (H, W)
                img = np.squeeze(img)

            # Final conversion to PIL with explicit mode
            if img.ndim == 2:
                img_pil = Image.fromarray(img, mode="L")
            elif img.ndim == 3 and img.shape[2] == 3:
                img_pil = Image.fromarray(img, mode="RGB")
            elif img.ndim == 3 and img.shape[2] == 4:
                img_pil = Image.fromarray(img, mode="RGBA")
            else:
                raise ValueError(
                    f"Unsupported image shape for saving: {img.shape}. "
                    "Expected (H, W), (H, W, 3), or (H, W, 4)."
                )

            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))
        
        labels_df = pd.DataFrame(
            [(k, v) for d in labels for k, v in d.items()],
            columns=['Image', 'Target']
        )
        labels_df.to_csv(os.path.join(save_dir, "classes.csv"))