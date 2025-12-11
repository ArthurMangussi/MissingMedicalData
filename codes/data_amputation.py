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
        foreground_mask_2d = np.any(x_data >= 0.01, axis=-1)

        missing_mask_2d = np.stack(
            (np.random.choice([0, 1], size=(x_data.shape[0], x_data.shape[1], x_data.shape[2]),
                            p=[1 - self.missing_rate, self.missing_rate]),) * number_channels, axis=-1)
        
        missing_mask_limited_2d = np.squeeze(missing_mask_2d, axis=-1) * foreground_mask_2d
        missing_mask = np.stack((missing_mask_limited_2d,) * number_channels, axis=-1)

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
        foreground_mask_2d = np.any(x_data >= 0.01, axis=-1)

        # Calcula a média dos canais (caso tenha mais de 1)
        grayscale = x_data.mean(axis=-1)  # Shape: (N, H, W)

        # Normaliza as intensidades para [0, 1]
        grayscale_norm = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min() + 1e-8)

        # Define as probabilidades de missing: quanto maior o valor, maior a probabilidade
        # multiplicado pelo missing_rate para manter taxa geral sob controle
        prob_missing = grayscale_norm * self.missing_rate

        # Gera máscara com base nas probabilidades
        missing_mask = np.random.binomial(1, prob_missing).astype(np.uint8)  # Shape: (N, H, W)

        missing_mask_limited_2d = missing_mask * foreground_mask_2d
        missing_mask = np.stack((missing_mask_limited_2d,) * number_channels, axis=-1)

        # Aplica missing
        x_data_md = x_data * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
        x_data_md[x_data_md == -1] = np.nan

        return x_data, x_data_md, missing_mask


    def generate_random_squares_mask(self,x_data: np.ndarray, square_size: int = 5) -> np.ndarray:
        """
        Gera uma máscara binária 2D com 'num_squares' quadrados de 'square_size' x 'square_size'.
        Os quadrados são posicionados aleatoriamente APENAS em pixels não-zero da imagem.

        Args:
            image_2d: Array NumPy 2D da imagem (H, W).
            num_squares: Número de quadrados a serem gerados (padrão é 4).
            square_size: Tamanho do lado do quadrado (padrão é 5).

        Returns:
            Um array NumPy 2D (H, W) representando a máscara binária (0s e 1s).
        """
        # Garante que a entrada seja 4D (N, H, W, C)
        if len(x_data.shape) == 3:
            x_data = np.expand_dims(x_data, axis=-1)

        x_data = x_data.astype('float32') / 255
        
        N, H, W, C = x_data.shape 
        
        # 1. Inicializar a lista para armazenar as máscaras 2D de cada imagem
        all_missing_masks_2d = []

        # 2. Iterar sobre cada imagem no batch
        for i in range(N):
            
            # C. Inicializar a máscara 2D para esta imagem
            mask_2d = np.zeros((H, W), dtype=np.uint8)

            start_x = (W - square_size) // 2
            end_x = start_x + square_size
            start_y = (H - square_size) // 2
            end_y = start_y + square_size
            
            # E. Aplicar o quadrado NA MÁSCARA 2D ATUAL
            mask_2d[start_y:end_y, start_x:end_x] = 1
        
            # Adicionar a máscara 2D gerada à lista
            all_missing_masks_2d.append(mask_2d)

        # 3. Empilhar as máscaras 2D de volta em um array 3D (N, H, W)
        missing_mask_3d = np.stack(all_missing_masks_2d, axis=0)

        # 4. Expansão para Canais e Aplicação da Falta (Como no seu código original)
        # Transforma a máscara (N, H, W) em (N, H, W, C) para multiplicação
        # (Adiciona a dimensão do canal para broadcasting)
        missing_mask_4d = np.expand_dims(missing_mask_3d, axis=-1)
        missing_mask_4d = np.repeat(missing_mask_4d, C, axis=-1)
        
        # Aplica missing: onde a máscara é 1, o valor será -1.0 temporariamente
        x_data_md = x_data * (~missing_mask_4d.astype(bool)).astype(x_data.dtype) + -1.0 * missing_mask_4d

        # Converte o -1.0 para np.nan. Isso requer que x_data_md seja float.
        # Se x_data for int, você precisa primeiro converter a saída para float:
        if x_data.dtype.kind in np.typecodes['AllInteger']:
            x_data_md = x_data_md.astype(np.float32)

        x_data_md[x_data_md == -1] = np.nan

        # Retornamos o x_data original, o corrompido, e a máscara 3D (N, H, W) se for usada no loss
        return x_data, x_data_md, missing_mask_3d