import numpy as np 

class ImageDataAmputation:
    def __init__(self, missing_rate: float):
        self.missing_rate = missing_rate
    
    def generate_missing_mask_mcar(self,x_data: np.ndarray):
        """
        Function to generate missing values in 2D images.

        Args:
            missing_rate: float, taxa de missing data
            img: np.ndarray, imagem 2D

        Returns:
            x_data_md: np.ndarray, imagem 2D com missing data
        """
        # missing_mask = np.random.choice([0, 1], size=(img.shape[0], img.shape[1]),
        #                         p=[1 - self.missing_rate, self.missing_rate])

        # x_data_md = img * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
        # x_data_md[x_data_md == -1] = np.nan

        # return x_data_md, missing_mask
        if len(x_data.shape) == 3:
            x_data = np.expand_dims(x_data, axis=3)
        x_data = x_data.astype('float32') / 255

        number_channels = x_data.shape[3]
        missing_mask = np.stack(
            (np.random.choice([0, 1], size=(x_data.shape[0], x_data.shape[1], x_data.shape[2]),
                            p=[1 - self.missing_rate, self.missing_rate]),) * number_channels, axis=-1)

        x_data_md = x_data * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
        x_data_md[x_data_md == -1] = np.nan

        return x_data, x_data_md, missing_mask
