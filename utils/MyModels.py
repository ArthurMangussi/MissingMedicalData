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

from algorithms.wrappers import KNNWrapper, MICEWrapper, MCWrapper


import numpy as np
from tensorflow import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


from keras.layers import Activation



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
    def model_knn():
        knn = KNNWrapper(n_neighbors=3)
        return knn
    # ------------------------------------------------------------------------
    def model_mice():
        mice = MICEWrapper(max_iter=10)
        return mice
    # ------------------------------------------------------------------------
    def model_mc():
        mc = MCWrapper()
        return mc
    # ------------------------------------------------------------------------
    def choose_model(self,
                     model: str, 
                     x_train: np.ndarray = None,
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
            case "knn":
                self._logger.info("[KNN] Training...")
                return ModelsImputation.model_knn()
            
            case "mice":
                self._logger.info("[MICE] Training...")
                return ModelsImputation.model_mice()
            
            case "mc":
                self._logger.info("[MC] Training...")
                return  ModelsImputation.model_mc()
            
class CNN:
    def __init__(self, 
                 img_shape,
                 batch_size:int = 32,
                 num_classes:int = 2,
                 learning_rate:float = 0.001,
                 epochs:int = 100):

        self.img_width, self.img_height = img_shape[0], img_shape[1]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self._model_cnn()
    
    def _model_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding ="same", input_shape=(self.img_width, self.img_height, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3,3), padding ="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  #classes_num or 2  
        return model

    
    def fit(self,
            x_train,
            y_train,
            x_val,
            y_val):
        train_datagen = ImageDataGenerator(rotation_range=180, 
                                           zoom_range=0.2, 
                                           shear_range=10, 
                                           horizontal_flip=True, 
                                           vertical_flip=True, 
                                           fill_mode="reflect")

        x_train_reshaped = np.expand_dims(x_train, axis=-1)
        x_val_reshaped = np.expand_dims(x_val, axis=-1)

        train_generator = train_datagen.flow(x_train_reshaped, y_train, batch_size=self.batch_size)
        validation_generator = train_datagen.flow(x_val_reshaped, y_val)

        # Early stopping (stop training after the validation loss reaches the minimum)
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1)

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(
                train_generator,
                steps_per_epoch=len(y_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=validation_generator,
                callbacks=[earlystopping])
        
    def predict(self,
                x_test):    
        predict = self.model.predict(x_test,
                                     batch_size =1)
        
        return predict