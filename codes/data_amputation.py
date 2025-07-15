import numpy as np 

class ImageDataAmputation:
    def __init__(self, missing_rate: float):
        self.missing_rate = missing_rate
    
    def generate_missing_mask_mcar(self,x_data: np.ndarray):
        """
        Function to generate missing values completely at 
        random in 2D images.

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

    def generate_missing_mask_mnar(self, x_data: np.ndarray):
        """
        Function to generate missing pixels not at random, 
        using the highest grayscale values in 2D images

        Args:
            missing_rate: float, taxa de missing data
            img: np.ndarray, imagem 2D

        Returns:
            x_data_md: np.ndarray, imagem 2D com missing data

        """
        if len(x_data.shape) == 3:
            x_data = np.expand_dims(x_data, axis=3)
    
        x_data = x_data.astype('float32') / 255
        number_channels = x_data.shape[3]

        # Calcula a média dos canais (caso tenha mais de 1)
        grayscale = x_data.mean(axis=-1)  # Shape: (N, H, W)

        # Normaliza as intensidades para [0, 1]
        grayscale_norm = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min() + 1e-8)

        # Define as probabilidades de missing: quanto maior o valor, maior a probabilidade
        # multiplicado pelo missing_rate para manter taxa geral sob controle
        prob_missing = grayscale_norm * self.missing_rate

        # Gera máscara com base nas probabilidades
        missing_mask = np.random.binomial(1, prob_missing).astype(np.uint8)  # Shape: (N, H, W)

        # Estende para todos os canais
        missing_mask = np.stack([missing_mask] * number_channels, axis=-1)  # Shape: (N, H, W, C)

        # Aplica missing
        x_data_md = x_data * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
        x_data_md[x_data_md == -1] = np.nan

        return x_data, x_data_md, missing_mask


        