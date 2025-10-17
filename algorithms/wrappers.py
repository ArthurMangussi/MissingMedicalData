from fancyimpute import KNN, IterativeImputer, IterativeSVD
import numpy as np
from algorithms.cvae import CVAE

class KNNWrapper:
    
    def __init__(self, 
                 n_neighbors:int):
        self.imputer = KNN(n_neighbors)
        
    def transform(self, images_with_mv_test):
        #Impute the data in each image
        images = []
        for k in range(images_with_mv_test.shape[0]):
            test_image = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.imputer.fit_transform(test_image)
            images.append(imputed_image)
            
        return np.array(images)
        
class MICEWrapper:
    
    # TODO: Must be improved to avoid unused parameters... Maybe use tuples?
    def __init__(self, 
                 max_iter:int):
        
        self.mice_impute = IterativeImputer(max_iter=max_iter,
                                            random_state=42)
        
    def transform(self, images_with_mv_test):
        #Impute the data in each image
        images = []
        for k in range(images_with_mv_test.shape[0]):
            test_image = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.mice_impute.fit_transform(test_image)
            images.append(imputed_image)
            
        return np.array(images)
    
class MCWrapper:
    
    def __init__(self):
        
        self.solver = IterativeSVD(rank=3)
        
    def transform(self, images_with_mv_test, masks_test):
        #Impute the data in each image
        imputed_images = []
        for k in range(images_with_mv_test.shape[0]):
            if(k%20 == 0):
                print("MC: Imputing Image " + str(k) + " of " + str(images_with_mv_test.shape[0]))
            mask_image = np.reshape(masks_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2],1))
            image_with_mv = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.solver.fit_transform(image_with_mv)
            imputed_images.append(imputed_image)
        
            
        return np.array(imputed_images)
    
class VAEWrapper:

    def __init__(self, 
                 images_train_val, 
                 images_with_mv_train_val, 
                 masks_train_val, 
                 ):
        
        custom_configurations = {
            "dense_hidden_layers_encoder": [392, 196],
            "dense_hidden_layers_decoder": [392],
            "latent_dim": 32,
            "reconstruction_missing_values_weight": 1,
            "kullback_leibler_weight": 1,
            "dropout_rate": 0.2,
            "l2_lambda": 0.01,
            "reduce_learning_rate_factor": 0.2,
            "reduce_learning_rate_patience": 10,
            "early_stopping_patience": 10,
            "epochs": 200,
            "validation_size": 0.1428
        }
        self.model = CVAE(custom_configurations, images_train_val, images_with_mv_train_val, masks_train_val)
        return

    def train(self):
        self.model.create_and_train()

    def transform(self, images_with_mv_test, masks_test):
        return self.model.predict([images_with_mv_test, masks_test])
