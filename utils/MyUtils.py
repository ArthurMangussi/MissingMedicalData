from PIL import Image
import os 
import numpy as np
import pandas as pd

class Utilities:
    @staticmethod
    def save_image(mechanism:str, 
                   images:np.ndarray, 
                   fold:int, 
                   model_impt:str,
                   labels_names:dict,
                   dataset:str,
                   image_ids:list=None,
                   ):
        """
        Method to save the array as an image.
        """
        save_dir = f"./new_results/{dataset}/{model_impt}/imputed_images/fold{fold}_{mechanism}"
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            labels = np.array([{str(i):labels_names[i]} for i in image_ids])
        except KeyError:
            image_ids = [i[:-4]for i in image_ids]
            labels = np.array([{str(i):labels_names[i]} for i in image_ids])

        
        for count, image in enumerate(images):
            # image shape here is (H, W) when images is (N, H, W)
            img = np.array(image, copy=True)

            # Normalize to [0, 255] uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Explicit grayscale conversion — avoids PIL KeyError
            # when ndim > 2 slips through (e.g. (H, W, 1) from some models)
            if img.ndim > 2:
                img = img.squeeze()

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))
        
        labels_df = pd.DataFrame(
            [(k, v) for d in labels for k, v in d.items()],
            columns=['Image', 'Target']
        )
        labels_df.to_csv(os.path.join(save_dir, "classes.csv"))

    @staticmethod
    def save_image_breakhist(
        mechanism: str,
        images: np.ndarray,
        missing_masks: np.ndarray,
        fold: int,
        model_impt: str,
        dataset: str,
        image_filenames,
        image_labels,
        image_patients,
    ):
        """
        Method to save the imputed BreaKHis images, together with the
        missing-pixel mask used to amputate each one, and a classes.csv that
        keeps full traceability (original filename, patient id and
        benigno/maligno label). The mask is required for a future
        presentation/regeneration of the incomplete image and for a
        downstream classification task -- both need the exact same
        IMG_XXXX/MASK_XXXX pairing.
        """
        save_dir = f"./new_results/{dataset}/{model_impt}/imputed_images/fold{fold}_{mechanism}"
        os.makedirs(save_dir, exist_ok=True)

        for count, image in enumerate(images):
            # image shape here is (H, W) when images is (N, H, W)
            img = np.array(image, copy=True)

            # Normalize to [0, 255] uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Explicit grayscale conversion — avoids PIL KeyError
            # when ndim > 2 slips through (e.g. (H, W, 1) from some models)
            if img.ndim > 2:
                img = img.squeeze()

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))

        for count, mask in enumerate(missing_masks):
            # 1 = missing, 0 = observed -> 255/0 for a viewable PNG mask
            mask_arr = np.array(mask, copy=True)
            if mask_arr.ndim > 2:
                mask_arr = mask_arr.squeeze()

            mask_img = (mask_arr.astype(bool).astype(np.uint8)) * 255
            mask_pil = Image.fromarray(mask_img, mode="L")
            mask_pil.save(os.path.join(save_dir, f"MASK_{count:04d}.png"))

        labels_df = pd.DataFrame(
            {
                "Image": [f"IMG_{i:04d}" for i in range(len(images))],
                "Mask": [f"MASK_{i:04d}" for i in range(len(images))],
                "Filename": list(image_filenames),
                "Patient": list(image_patients),
                "Target": list(image_labels),
            }
        )
        labels_df.to_csv(os.path.join(save_dir, "classes.csv"))

    @staticmethod
    def save_image_cbis(mechanism:str,
                   images:np.ndarray, 
                   fold:int, 
                   model_impt:str,
                   dataset:str,
                   ):
        """
        Method to save the array as an image.
        """
        save_dir = f"./new_results/{dataset}/{model_impt}/imputed_images/fold{fold}_{mechanism}"
        os.makedirs(save_dir, exist_ok=True)
                
        for count, image in enumerate(images):
            # image shape here is (H, W) when images is (N, H, W)
            img = np.array(image, copy=True)

            # Normalize to [0, 255] uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Explicit grayscale conversion — avoids PIL KeyError
            # when ndim > 2 slips through (e.g. (H, W, 1) from some models)
            if img.ndim > 2:
                img = img.squeeze()

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(save_dir, f"IMG_{count:04d}.png"))
