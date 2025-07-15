# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

import warnings

import numpy as np
import tensorflow as tf

# Variational Autoencoder with Weighted Loss
from algorithms.vae import ConfigVAE
from algorithms.vaewl import VAEWL
from utils.MeLogSingle import MeLogger

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

class ModelsImputation:
    def __init__(self) :
        self._logger = MeLogger()
    
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_vaewl(x_train:np.ndarray,
                    x_train_md:np.ndarray,
                    x_val_md: np.ndarray,
                    x_val:np.ndarray):
        
        vae_wl_config = ConfigVAE()
        vae_wl_config.verbose = 1
        vae_wl_config.epochs = 200
        vae_wl_config.filters = [32, 64]
        vae_wl_config.kernels = 3
        vae_wl_config.neurons = [392, 196]
        vae_wl_config.dropout = [0.2, 0.2]
        vae_wl_config.latent_dimension = 32
        vae_wl_config.batch_size = 64
        vae_wl_config.learning_rate = 0.001
        vae_wl_config.activation = "relu"
        vae_wl_config.output_activation = "sigmoid"
        vae_wl_config.loss = tf.keras.losses.binary_crossentropy
        vae_wl_config.input_shape = x_train.shape[1:]
        vae_wl_config.missing_values_weight = 5
        vae_wl_config.kullback_leibler_weight = 0.1

        vae_wl_model = VAEWL(vae_wl_config)
        vaewl = vae_wl_model.fit(x_train_md, 
                                 x_train, 
                                 X_val=x_val_md, 
                                 y_val=x_val)
        
        return vaewl
    
    # ------------------------------------------------------------------------
    def choose_model(self,
                     model: str, 
                     x_train: np.ndarray,
                     x_train_md: np.ndarray = None,
                       x_val_md: np.ndarray = None, 
                       x_val: np.ndarray = None):
        match model:

            case "vaewl":
                self._logger.info("[VAE-WL] Training...")
                return ModelsImputation.model_vaewl(x_train=x_train,
                                                    x_train_md=x_train_md,
                                                    x_val_md=x_val_md,
                                                    x_val=x_val)