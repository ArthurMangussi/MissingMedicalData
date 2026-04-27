from fancyimpute import KNN, IterativeImputer, IterativeSVD
import numpy as np


class KNNWrapper:
    """
    KNN-based imputation wrapper for multi-channel images.

    Handles 2D grayscale and multi-channel images by flattening spatial dimensions
    while preserving the original shape in the output.
    """

    def __init__(self, n_neighbors: int):
        """
        Initialize KNN imputer.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors for KNN imputation
        """
        self.imputer = KNN(n_neighbors)

    def transform(self, images_with_mv_test):
        """
        Impute missing values in a batch of images.

        Supports arbitrary input shapes: (N, H, W), (N, C, H, W), (N, H, W, C)

        Parameters
        ----------
        images_with_mv_test : np.ndarray
            Batch of images with missing values (NaN)
            Shape: (N, H, W) or (N, C, H, W) or (N, H, W, C)

        Returns
        -------
        imputed_images : np.ndarray
            Imputed images with same shape as input
        """
        images_with_mv_test = np.squeeze(images_with_mv_test)  

        imputed_images = []
        for k in range(images_with_mv_test.shape[0]):
            if(k%100 == 0):
                print("KNN: Imputing Image " + str(k) + " of " + str(images_with_mv_test.shape[0]))
            test_image = np.reshape(images_with_mv_test[k],(images_with_mv_test.shape[1],images_with_mv_test.shape[2]))
            imputed_image = self.imputer.fit_transform(test_image)
            imputed_images.append(imputed_image)

        return np.array(imputed_images)


class MCWrapper:

    def __init__(self):

        self.solver = IterativeSVD(rank=3)

    def transform(self, images_with_mv_test, masks_test):
        # Impute the data in each image
        imputed_images = []
        for k in range(images_with_mv_test.shape[0]):
            if k % 20 == 0:
                print(
                    "MC: Imputing Image "
                    + str(k)
                    + " of "
                    + str(images_with_mv_test.shape[0])
                )
            mask_image = np.reshape(
                masks_test[k],
                (images_with_mv_test.shape[1], images_with_mv_test.shape[2], 1),
            )
            image_with_mv = np.reshape(
                images_with_mv_test[k],
                (images_with_mv_test.shape[1], images_with_mv_test.shape[2]),
            )
            imputed_image = self.solver.fit_transform(image_with_mv)
            imputed_images.append(imputed_image)

        return np.array(imputed_images)
