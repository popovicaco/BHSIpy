import numpy as np


class ErrorAnalyzer:
    """
    Class for Analyzing the Error of Linear Hyperspectral Unmixing Methods

    Input: Hyperspectral Noise Matrix w (nx, ny, n_bands)
    """

    def __init__(self, w):
        self.w = w
        self.nx, self.ny, self.n_bands = w.shape

    def average_LSE(self) -> float:
        '''
        Calculates the average SSR of a hyperspectral noise matrix pixelwise
        '''
        lse_matrix = np.zeros(shape=(self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                lse_matrix[i, j] = np.sum(self.w[i, j, :] ** 2)

        return np.sum(lse_matrix) / (self.nx * self.ny)
