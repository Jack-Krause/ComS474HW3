import numpy as np
'''
Homework5: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.eig(): compute the eigen decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

    def __init__(self, X, n_components):
        """
        Args:
            X: The data matrix of shape [n_samples, n_features].
            n_components: The number of principal components. A scaler number.
        """

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()

    
    def _do_pca(self):
        """
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        """
        ### YOUR CODE HERE
        print(f"X: {self.X[0][:10]}")
        print(np.shape(self.X))
        # X_bar = np.mean(self.X)
        # X_bar = round(np.mean(self.X, keepdims=True), 4)
        # 1. find mean
        X_bar = np.mean(self.X, keepdims=True, axis=0)

        # 2. center the data
        centered_X = self.X - X_bar

        # 3. calculate the covariance of the centered data
        covariance_X = np.cov(centered_X, rowvar=False)

        # 4. perform eigendecomposition (vectors need to be sorted by values)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_X)

        # sort the eigenvectors (desc) by corresponding eigenvalues
        idx = np.argsort(eigenvalues)[:: -1]
        eigenvectors = eigenvectors[:, idx]

        # 5. The transition matrix is Up: (n_features x n_components)
        self.Up = eigenvectors[:, :self.n_components]

        # 6. apply the transition matrix on centered X
        self.Xp = centered_X @ self.Up
        new_X = (np.transpose(self.Up) @ centered_X)

        # 7. reapply the transition matrix
        reconstructed_centered = new_X @ self.Up

        # 8. add back the mean
        reconstructed = reconstructed_centered + X_bar

        return self.Up, self.Xp
        ### END YOUR CODE

    def get_reduced(self):
        """
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_any, n_features] or None.
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_any, n_components].
        """
        return self.Xp

    def reconstruction(self, Xp):
        """
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        Return:
        X_re: The reconstructed matrix of shape [n_samples, n_features].
        """
        ### YOUR CODE HERE



        ### END YOUR CODE


def reconstruct_error(A, B):
    """
    To compute the reconstruction error.

    Args:
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return:
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    """
    ### YOUR CODE HERE
    return np.linalg.norm(A - B, ord='fro')**2


    ### END YOUR CODE

