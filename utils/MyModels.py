# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = "Arthur Dantas Mangussi"

import warnings

import numpy as np
import tensorflow as tf

# Variational Autoencoder with Weighted Loss
from algorithms.vae import ConfigVAE
from algorithms.vaewl import VAEWL
from utils.MeLogSingle import MeLogger

from algorithms.wrappers import KNNWrapper, MCWrapper
import algorithms.dip_code as dip

import numpy as np
from tensorflow import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# from algorithms.mae_vit import MAE
# from algorithms.vit import ViT

from keras.layers import Activation
from algorithms import models_mae
import torch
from PIL import Image

# Diffusion-based inpainting
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


class DeepImagePrior:
    """
    Deep Image Prior (DIP) for image imputation using untrained neural networks.

    The DIP approach leverages the implicit regularization of an untrained convolutional
    network architecture to perform image restoration tasks, including inpainting/imputation
    of missing pixels. The network learns to reconstruct only the observed pixels.

    Reference: https://github.com/DmitryUlyanov/deep-image-prior
    """

    def __init__(
        self,
        device: str = "cuda",
        num_iter: int = 400,
        learning_rate: float = 0.03,
        input_depth: int = 32,
        num_channels: int = 128,
        pad: str = "reflection",
        reg_noise_std: float = 0.03,
        show_every: int = 50,
    ):
        """
        Initialize Deep Image Prior model.

        Parameters
        ----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Default: 'cuda'
        num_iter : int, optional
            Number of optimization iterations. Default: 400
        learning_rate : float, optional
            Learning rate for Adam optimizer. Default: 0.03
        input_depth : int, optional
            Number of input channels for noise. Default: 32
        num_channels : int, optional
            Number of channels in skip network. Default: 128
        pad : str, optional
            Padding mode ('zero' or 'reflection'). Default: 'reflection'
        reg_noise_std : float, optional
            Standard deviation of regularization noise. Default: 0.03
        show_every : int, optional
            Print loss every N iterations. Default: 50
        """
        self.device = device
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.input_depth = input_depth
        self.num_channels = num_channels
        self.pad = pad
        self.reg_noise_std = reg_noise_std
        self.show_every = show_every
        self.net = None
        self.losses = []

        # Set torch device
        if device == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def fit(
        self,
        x_train: np.ndarray,
        mask_train: np.ndarray,
    ) -> np.ndarray:
        """
        Train DIP network on image with missing pixels.

        Parameters
        ----------
        x_train : np.ndarray
            Image with missing pixels (should be normalized to [0,1] or uint8).
            Shape: (H, W) or (C, H, W)
        mask_train : np.ndarray
            Binary mask indicating missing pixels (1 = missing, 0 = present).
            Shape must match x_train

        Returns
        -------
        imputed_image : np.ndarray
            Reconstructed image with missing pixels filled.
            Shape: same as x_train
        """
        # Ensure proper shape for processing
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=0)  # Add channel dimension

        if len(mask_train.shape) == 2:
            mask_train = np.expand_dims(mask_train, axis=0)

        # Normalize if needed
        if x_train.max() > 1.0:
            x_train = x_train.astype(np.float32) / 255.0

        # Prepare network and input
        num_channels = x_train.shape[0]
        spatial_size = x_train.shape[1:]

        # Build skip network
        self.net = dip.skip(
            num_input_channels=self.input_depth,
            num_output_channels=num_channels,
            num_channels_down=[self.num_channels] * 5,
            num_channels_up=[self.num_channels] * 5,
            num_channels_skip=[self.num_channels] * 5,
            filter_size_up=3,
            filter_size_down=3,
            upsample_mode="nearest",
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            pad=self.pad,
            act_fun="LeakyReLU",
        ).type(self.dtype)

        # Generate noise input
        net_input = dip.get_noise(self.input_depth, "noise", spatial_size).type(
            self.dtype
        )

        # Convert to torch tensors
        img_var = dip.np_to_torch(x_train).type(self.dtype)
        mask_var = dip.np_to_torch(mask_train).type(self.dtype)

        # Loss function
        mse = torch.nn.MSELoss().type(self.dtype)

        # Save original input for regularization
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        # Optimization parameters
        params = dip.get_params("net", self.net, net_input)

        # Define closure for optimization
        def closure():
            if self.reg_noise_std > 0:
                net_input_current = net_input_saved + (
                    noise.normal_() * self.reg_noise_std
                )
            else:
                net_input_current = net_input_saved

            out = self.net(net_input_current)

            # Compute loss only on observed pixels
            total_loss = mse(out * mask_var, img_var * mask_var)
            total_loss.backward()

            return total_loss

        # Run optimization
        dip.optimize("adam", params, closure, self.learning_rate, self.num_iter)

        # Get final reconstruction
        with torch.no_grad():
            reconstructed = self.net(net_input_saved).detach()

        imputed_np = dip.torch_to_np(reconstructed)

        # Clean up GPU memory
        torch.cuda.empty_cache()

        return imputed_np

    def transform(self, x_test: np.ndarray) -> np.ndarray:
        """
        Apply trained DIP network to test image.

        Note: DIP is typically used as a single-image method without pre-training.
        This method applies the trained network if available.

        Parameters
        ----------
        x_test : np.ndarray
            Image to process

        Returns
        -------
        output : np.ndarray
            Processed image
        """
        if self.net is None:
            raise RuntimeError("Network not trained. Call fit() first.")

        if len(x_test.shape) == 2:
            x_test = np.expand_dims(x_test, axis=0)

        if x_test.max() > 1.0:
            x_test = x_test.astype(np.float32) / 255.0

        x_test_torch = dip.np_to_torch(x_test).type(self.dtype)

        with torch.no_grad():
            output = self.net(x_test_torch).detach()

        return dip.torch_to_np(output)

    def fit_transform(
        self,
        x_train: np.ndarray,
        mask_train: np.ndarray,
    ) -> np.ndarray:
        """
        Fit DIP on training image and return imputed result.

        Convenience method combining fit() and direct output.

        Parameters
        ----------
        x_train : np.ndarray
            Image with missing pixels
        mask_train : np.ndarray
            Binary mask of missing pixels

        Returns
        -------
        imputed_image : np.ndarray
            Reconstructed image
        """
        return self.fit(x_train, mask_train)


class DiffusionInpainting:
    """
    Diffusion Model for image inpainting using pre-trained HuggingFace pipelines.

    The diffusion model approach uses a pre-trained generative model fine-tuned for
    inpainting tasks. It iteratively denoises a random image conditioned on observed
    pixels to reconstruct missing regions naturally.

    Supports medical imaging models like `Likalto4/vindr_lesion-inpainting` for
    lesion and artifact inpainting in X-ray and mammography images.

    Reference: https://huggingface.co/docs/diffusers
    """

    def __init__(
        self,
        model_dir: str = "Likalto4/inpainting_vindr_massbs16",
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        """
        Initialize Diffusion Inpainting model.

        Parameters
        ----------
        model_dir : str, optional
            HuggingFace model identifier or local path. Default: `Likalto4/vindr_lesion-inpainting`
        device : str, optional
            Device to use ('cuda' or 'cpu'). Default: 'cuda'
        torch_dtype : torch.dtype, optional
            Data type for model (float16 for memory efficiency, float32 for precision). Default: float16
        """
        self.model_dir = model_dir
        self.device = device
        self.torch_dtype = torch_dtype

        # Load diffusion pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)

        # Set scheduler for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Enable memory-efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            # xformers may not be installed, but pipeline will still work
            pass

    def _to_pil(self, arr: np.ndarray) -> tuple:
        """
        Convert numpy array to PIL Image(s) for pipeline.

        Handles:
        - Grayscale (H, W) → RGB PIL
        - Single channel (1, H, W) → RGB PIL
        - Multi-channel (3, H, W) → RGB PIL
        - Uint8 or float [0,1] normalization

        Returns
        -------
        pil_image : PIL.Image.Image
            RGB PIL image
        """
        # Normalize if needed
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0

        # Handle shape variations
        if len(arr.shape) == 2:
            # (H, W) grayscale
            arr_rgb = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
        elif len(arr.shape) == 3:
            if arr.shape[0] == 1:
                # (1, H, W) → (3, H, W)
                arr_rgb = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] == 3:
                # Already RGB (3, H, W)
                arr_rgb = arr
            elif arr.shape[2] == 1:
                # (H, W, 1) → (H, W, 3)
                arr_rgb = np.repeat(arr, 3, axis=2)
                arr_rgb = arr_rgb.transpose(2, 0, 1)  # → (3, H, W)
            elif arr.shape[2] == 3:
                # (H, W, 3) → (3, H, W)
                arr_rgb = arr.transpose(2, 0, 1)
            else:
                raise ValueError(f"Unsupported shape: {arr.shape}")
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

        # Convert to PIL (expects [0, 255] uint8)
        arr_uint8 = np.clip(arr_rgb * 255, 0, 255).astype(np.uint8)
        # PIL expects (H, W, C)
        arr_hwc = arr_uint8.transpose(1, 2, 0)
        return Image.fromarray(arr_hwc, mode="RGB")

    def _to_np(
        self, pil_image: Image.Image, original_shape: tuple
    ) -> np.ndarray:
        """
        Convert PIL Image back to numpy array matching original shape.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            RGB PIL image
        original_shape : tuple
            Target shape to match

        Returns
        -------
        arr : np.ndarray
            Numpy array [0, 1], shape matching original input
        """
        # Convert PIL to numpy (H, W, 3) [0, 255]
        arr_uint8 = np.array(pil_image)  # (H, W, 3)
        arr = arr_uint8.astype(np.float32) / 255.0

        # Match original shape
        if len(original_shape) == 2:
            # Return single grayscale channel
            return arr[:, :, 0]
        elif len(original_shape) == 3:
            if original_shape[0] == 1:
                # (1, H, W)
                return arr[:, :, 0:1].transpose(2, 0, 1)
            elif original_shape[0] == 3:
                # (3, H, W)
                return arr.transpose(2, 0, 1)
            else:
                # Return first channel
                return arr[:, :, 0:1].transpose(2, 0, 1)
        else:
            return arr

    def fit(
        self,
        x_train: np.ndarray,
        mask_train: np.ndarray,
        prompt: str = "medical image",
        num_inference_steps: int = 20,
    ) -> np.ndarray:
        """
        Apply diffusion inpainting to a single image with missing pixels.

        The diffusion model reconstructs missing pixels while preserving observed regions.

        Parameters
        ----------
        x_train : np.ndarray
            Image with missing pixels. Shape: (H, W), (C, H, W), or (H, W, C)
            Can be uint8 [0, 255] or float [0, 1]
        mask_train : np.ndarray
            Binary mask (1 = missing, 0 = present). Same shape conventions as x_train
        prompt : str, optional
            Text prompt for guided generation. Default: 'medical image'
        num_inference_steps : int, optional
            Number of diffusion steps (higher = better quality but slower). Default: 20

        Returns
        -------
        imputed_image : np.ndarray
            Reconstructed image with missing pixels filled
        """
        original_shape = x_train.shape

        # Normalize mask to [0, 1]
        if mask_train.max() > 1.0:
            mask_train = mask_train.astype(np.float32) / 255.0

        # Convert to PIL images
        image_pil = self._to_pil(x_train)
        mask_pil = self._to_pil(mask_train)

        # Invert mask if needed (diffusion typically uses 1=inpaint, 0=keep)
        # User convention: 1=missing, 0=present → invert for pipeline
        mask_pil = Image.fromarray(
            255 - np.array(mask_pil).astype(np.uint8), mode="RGB"
        )

        # Run diffusion inpainting
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
            )

        output_pil = output.images[0]
        imputed_np = self._to_np(output_pil, original_shape)

        # Blend: preserve observed pixels exactly
        mask_train_normalized = mask_train
        if len(mask_train.shape) != len(x_train.shape):
            if len(x_train.shape) == 3 and len(mask_train.shape) == 2:
                mask_train_normalized = np.expand_dims(mask_train, axis=0)

        # Normalize x_train if needed
        x_train_normalized = x_train
        if x_train.max() > 1.0:
            x_train_normalized = x_train.astype(np.float32) / 255.0

        # Blend
        blended = x_train_normalized * (1 - mask_train_normalized) + imputed_np * mask_train_normalized

        return blended

    def transform(
        self,
        x_test: np.ndarray,
        mask_test: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Apply trained diffusion model to test image.

        Diffusion models are always used for inference (no separate training step).
        This method applies inpainting to new images.

        Parameters
        ----------
        x_test : np.ndarray
            Test image
        mask_test : np.ndarray
            Mask indicating missing pixels
        **kwargs : optional
            Additional arguments for fit() (prompt, num_inference_steps, etc.)

        Returns
        -------
        output : np.ndarray
            Imputed image
        """
        return self.fit(x_test, mask_test, **kwargs)

    def fit_transform(
        self,
        x_train: np.ndarray,
        mask_train: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Fit and transform in one call (delegates to fit).

        Parameters
        ----------
        x_train : np.ndarray
            Image with missing pixels
        mask_train : np.ndarray
            Missing mask
        **kwargs : optional
            Additional arguments

        Returns
        -------
        imputed_image : np.ndarray
            Reconstructed image
        """
        return self.fit(x_train, mask_train, **kwargs)


class ModelsImputation:
    def __init__(self):
        self._logger = MeLogger()

    # ------------------------------------------------------------------------
    @staticmethod
    def model_vaewl(
        x_train: np.ndarray,
        x_train_md: np.ndarray,
        x_val_md: np.ndarray,
        x_val: np.ndarray,
    ):

        vae_wl_config = ConfigVAE()
        vae_wl_config.verbose = 1
        vae_wl_config.epochs = 100
        vae_wl_config.filters = [16, 32]
        vae_wl_config.kernels = 1
        vae_wl_config.neurons = [392, 196]
        vae_wl_config.dropout = [0.2, 0.2]
        vae_wl_config.latent_dimension = 16
        vae_wl_config.batch_size = 16
        vae_wl_config.learning_rate = 0.0005
        vae_wl_config.activation = "relu"
        vae_wl_config.output_activation = "sigmoid"
        vae_wl_config.loss = tf.keras.losses.binary_crossentropy
        vae_wl_config.input_shape = x_train.shape[1:]
        vae_wl_config.missing_values_weight = 100
        vae_wl_config.kullback_leibler_weight = 0.01

        vae_wl_model = VAEWL(vae_wl_config)
        vaewl = vae_wl_model.fit(x_train_md, x_train, X_val=x_val_md, y_val=x_val)

        return vaewl

    # ------------------------------------------------------------------------
    @staticmethod
    def model_mae_vit(chkpt_dir, arch="mae_vit_large_patch16"):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location="cuda")
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
        return model

    # ------------------------------------------------------------------------
    @staticmethod
    def model_dip(
        num_iter: int = 400,
        learning_rate: float = 0.03,
        input_depth: int = 32,
        num_channels: int = 128,
        device: str = "cuda",
    ):
        """
        Initialize a Deep Image Prior model with specified hyperparameters.

        Parameters
        ----------
        num_iter : int, optional
            Number of optimization iterations. Default: 400
        learning_rate : float, optional
            Learning rate for Adam optimizer. Default: 0.03
        input_depth : int, optional
            Number of input channels for noise. Default: 32
        num_channels : int, optional
            Number of channels in skip network. Default: 128
        device : str, optional
            Device to use ('cuda' or 'cpu'). Default: 'cuda'

        Returns
        -------
        dip_model : DeepImagePrior
            Initialized DIP model ready for fitting
        """
        model =DeepImagePrior(
            device=device,
            num_iter=num_iter,
            learning_rate=learning_rate,
            input_depth=input_depth,
            num_channels=num_channels,
        )
        model.fit(x_train=x_train, mask_train=mask_train)
        return model

    # ------------------------------------------------------------------------
    @staticmethod
    def model_diffusion(
        model_dir: str = "Likalto4/inpainting_vindr_massbs16",
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        """
        Initialize a Diffusion-based inpainting model.

        Parameters
        ----------
        model_dir : str, optional
            HuggingFace model identifier or local path. Default: `Likalto4/vindr_lesion-inpainting`
            (Medical imaging lesion inpainting model)
        device : str, optional
            Device to use ('cuda' or 'cpu'). Default: 'cuda'
        torch_dtype : torch.dtype, optional
            Data type for model (float16 for memory efficiency, float32 for precision). Default: float16

        Returns
        -------
        diffusion_model : DiffusionInpainting
            Initialized diffusion inpainting model ready for inference
        """
        return DiffusionInpainting(
            model_dir=model_dir,
            device=device,
            torch_dtype=torch_dtype,
        )

    # ------------------------------------------------------------------------
    @staticmethod
    def model_knn():
        knn = KNNWrapper(n_neighbors=5)
        return knn

    # ------------------------------------------------------------------------
    @staticmethod
    def model_mc():
        mc = MCWrapper()
        return mc

    @staticmethod
    def mae_imputer_transform(model, x_test_md_np, missing_mask_test_np, missing_rate):
        """
        Adapta a lógica MAE para imputar o lote de teste x_test_md.
        """
        model.eval()  # Modo de avaliação (inferência)

        # 1. Preparação dos Tensores
        x = torch.tensor(x_test_md_np)
        x = x.repeat(1, 1, 1, 3)  # Dados incompletos
        if len(missing_mask_test_np.shape) == 3:
            missing_mask_test_np = np.expand_dims(missing_mask_test_np, axis=-1)
        mask_ext = torch.tensor(missing_mask_test_np)  # Máscara de ausência

        x = x.float()  # MAE espera float32
        x = torch.einsum("nhwc->nchw", x)
        mask_ext = mask_ext.float()
        mask_ext = torch.einsum("nhwc->nchw", mask_ext)

        # 3. Execução do MAE (Modo Imputação)
        with torch.no_grad():
            loss, y_reconstructed, mask_int = model(x, mask_ratio=0.75)

            # 4. Processamento da Reconstrução
            y_recon = model.unpatchify(y_reconstructed)

            # Transposição de volta para NHWC
            y_recon_nhwc = torch.einsum("nchw->nhwc", y_recon).cpu().numpy()
            x_nhwc = torch.einsum("nchw->nhwc", x).cpu().numpy()
            mask_ext_nhwc = torch.einsum("nchw->nhwc", mask_ext).cpu().numpy()

            imputed_image = x_nhwc * (1 - mask_ext_nhwc) + y_recon_nhwc * mask_ext_nhwc
            imputed_image_gray = np.mean(imputed_image, axis=-1, keepdims=True)

        return imputed_image_gray

    @staticmethod
    def diffusion_transform(
        model,
        x_test_md_np: np.ndarray,
        missing_mask_test_np: np.ndarray,
        prompt: str = "medical image",
        num_inference_steps: int = 20,
    ):
        """
        Apply diffusion inpainting to a batch of images.

        Processes each image individually (diffusion models typically work on single images).

        Parameters
        ----------
        model : DiffusionInpainting
            Trained diffusion inpainting model
        x_test_md_np : np.ndarray
            Batch of images with missing pixels. Shape: (N, H, W) or (N, C, H, W)
        missing_mask_test_np : np.ndarray
            Batch of binary masks. Shape must match x_test_md_np
        prompt : str, optional
            Text prompt for guided generation. Default: 'medical image'
        num_inference_steps : int, optional
            Number of diffusion steps. Default: 20

        Returns
        -------
        imputed_batch : np.ndarray
            Imputed images with same shape as input
        """
        # Handle batch dimension
        if len(x_test_md_np.shape) == 3:
            # (N, H, W) grayscale batch
            imputed_list = []
            for i in range(x_test_md_np.shape[0]):
                imputed = model.fit(
                    x_test_md_np[i],
                    missing_mask_test_np[i],
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                )
                imputed_list.append(imputed)
            return np.stack(imputed_list, axis=0)

        elif len(x_test_md_np.shape) == 4:
            # (N, C, H, W) multi-channel batch
            imputed_list = []
            for i in range(x_test_md_np.shape[0]):
                imputed = model.fit(
                    x_test_md_np[i],
                    missing_mask_test_np[i],
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                )
                imputed_list.append(imputed)
            return np.stack(imputed_list, axis=0)

        else:
            # Single image
            return model.fit(
                x_test_md_np,
                missing_mask_test_np,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
            )

    # ------------------------------------------------------------------------
    def choose_model(
        self,
        model: str,
        x_train: np.ndarray = None,
        x_train_md: np.ndarray = None,
        x_val_md: np.ndarray = None,
        x_val: np.ndarray = None
    ):
        match model:

            case "vaewl":
                self._logger.info("[VAE-WL] Training...")
                return ModelsImputation.model_vaewl(
                    x_train=x_train,
                    x_train_md=x_train_md,
                    x_val_md=x_val_md,
                    x_val=x_val,
                )
            case "knn":
                self._logger.info("[KNN] Training...")
                return ModelsImputation.model_knn()


            case "mc":
                self._logger.info("[MC] Training...")
                return ModelsImputation.model_mc()

            case "mae-vit":
                # Loop de treinamento para o MAE-ViT
                self._logger.info("[MAE-ViT] Importing...")
                model_pth = "/home/gpu-10-2025/Área de trabalho/Modelos/mae_visualize_vit_large.pth"
                return ModelsImputation.model_mae_vit(model_pth)

            case "mae-vit-gan":
                # Loop de treinamento para o MAE-ViT
                self._logger.info("[MAE-ViT] Importing...")
                model_pth = "/home/gpu-10-2025/Área de trabalho/Modelos/mae_visualize_vit_large_ganloss.pth"
                return ModelsImputation.model_mae_vit(model_pth)

            case "dip":
                self._logger.info("[DIP] Initializing...")
                return ModelsImputation.model_dip()

            case "diffusion":
                self._logger.info("[Diffusion] Loading pipeline...")
                return ModelsImputation.model_diffusion()


class CNN:
    def __init__(
        self,
        img_shape,
        batch_size: int = 32,
        num_classes: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ):

        self.img_width, self.img_height = img_shape[0], img_shape[1]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self._model_cnn()

    def _model_cnn(self):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                input_shape=(self.img_width, self.img_height, 1),
            )
        )
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))  # classes_num or 2
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        train_datagen = ImageDataGenerator(
            rotation_range=180,
            zoom_range=0.2,
            shear_range=10,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="reflect",
        )

        x_train_reshaped = np.expand_dims(x_train, axis=-1)
        x_val_reshaped = np.expand_dims(x_val, axis=-1)

        train_generator = train_datagen.flow(
            x_train_reshaped, y_train, batch_size=self.batch_size
        )
        validation_generator = train_datagen.flow(x_val_reshaped, y_val)

        # Early stopping (stop training after the validation loss reaches the minimum)
        earlystopping = EarlyStopping(
            monitor="val_loss", mode="min", patience=40, verbose=1
        )

        # Compile the model
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            train_generator,
            steps_per_epoch=len(y_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=[earlystopping],
        )

    def predict(self, x_test):
        predict = self.model.predict(x_test, batch_size=1)

        return predict
