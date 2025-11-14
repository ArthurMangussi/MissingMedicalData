from fancyimpute import KNN, IterativeImputer, IterativeSVD
import numpy as np


class KNNWrapper:

    def __init__(self, n_neighbors: int):
        self.imputer = KNN(n_neighbors)

    def transform(self, images_with_mv_test):
        # Impute the data in each image
        images = []
        for k in range(images_with_mv_test.shape[0]):
            test_image = np.reshape(
                images_with_mv_test[k],
                (images_with_mv_test.shape[1], images_with_mv_test.shape[2]),
            )
            imputed_image = self.imputer.fit_transform(test_image)
            images.append(imputed_image)

        return np.array(images)


class MICEWrapper:

    # TODO: Must be improved to avoid unused parameters... Maybe use tuples?
    def __init__(self, max_iter: int, x_train: np.ndarray):

        self.mice_impute = IterativeImputer(
            max_iter=max_iter,
            random_state=42,
            verbose=1,
            skip_complete=True,
            initial_strategy="median",
        )
        train_images = x_train.reshape((x_train.shape[0], -1))
        self.mice_impute.fit(X=train_images)

    def transform(self, images_with_mv_test):

        # Impute the data in each image
        test_images = images_with_mv_test.reshape((images_with_mv_test.shape[0], -1))
        imputed_image = self.mice_impute.transform(test_images)
        imputed_image = imputed_image.reshape(images_with_mv_test.shape)

        return np.array(imputed_image)


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
