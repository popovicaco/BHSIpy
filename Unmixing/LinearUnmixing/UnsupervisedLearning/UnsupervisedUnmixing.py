import numpy as np

class UnsupervisedLinearUnmixing:
    """
    Class for Unsupervised Linear Unmixing Methods
    Input: 
        Hyperspectral Image Output r: (nx, ny, n_bands)
    """

    def __init__(self, r: np.array):
        # Getting nx,ny and n_bands
        self.nx, self.ny, self.n_bands = r.shape
        # This is the desired output hyperspectral image cube
        self.r = r

    @staticmethod
    def __nnls(M: np.array, r: np.array) -> np.array:
        """
        Non-negative Least Squares using Langrangian Multipliers
        :param M: Spectral Endmember Matrix (n_bands, n_end)
        :param r: Pixel Vector (1, n_bands)
        :return x: Abundance Vector
        """
        r = r.reshape((-1, 1))
        # Get the number of end members
        n_end = M.shape[1]
        # Initialize set of nonactive columns to null
        P = np.full((n_end, 1), False)
        # Initialize set of active columns to all and the initial point x0 to zeros
        Z = np.full((n_end, 1), True)
        x = np.zeros(shape=(n_end, 1))
        # Calculating residuals
        residual = r - np.dot(M, x)
        w = np.dot(M.T, residual)

        # Tolerance Calculations
        eps = 2.22e-16
        tolerance = 10 * eps * np.linalg.norm(M, 1) * (max(M.shape))  # Why?

        # Iteration Criterion
        outer_iteration = 0
        iteration = 0
        max_iteration = 3 * n_end
        while np.any(Z) and np.any(w[Z] > tolerance):
            outer_iteration += 1
            # Reset intermediate solution
            z = np.zeros(shape=(n_end, 1))
            # Create wz a Lagrange multiplier vector of variables in the zero set
            wz = np.zeros(shape=(n_end, 1))
            wz[P] = np.NINF
            wz[Z] = w[Z]
            # Find index of variable with largest Lagrange multiplier, move it from zero set to positive set
            idx = wz.argmax()
            P[idx] = True
            Z[idx] = False
            # Compute the Intermediate Solution using only positive variables in P
            z[P.flatten()] = np.dot(np.linalg.pinv(M[:, P.flatten()]), r)
            # Inner loop to remove elements from the positive set that do not belong
            while np.any(z[P] <= 0):
                iteration += 1
                if iteration > max_iteration:
                    x = z
                    residual = r - np.dot(M, x)
                    w = np.dot(M.T, residual)
                    return x
                # Find indices where approximate solution is negative
                Q = (z <= 0) & P
                # Choose new x subject keeping it non-negative
                alpha = min(x[Q] / (x[Q] - z[Q]))
                x = x + alpha * (z - x)
                # Reset Z and P given intermediate values of x
                Z = ((abs(x) < tolerance) & P) | Z
                P = ~Z
                z = np.zeros(shape=(n_end, 1))
                # Resolve for z
                z[P.flatten()] = np.dot(np.linalg.pinv(M[:, P.flatten()]), r)
            x = np.copy(z)
            residual = r - np.dot(M, x)
            w = np.dot(M.T, residual)
        return x

    @staticmethod
    def __prep_fclsu(M: np.array, r: np.array, delta: float = 1e-3) -> tuple[np.array, np.array]:
        """
        Prepares an endmember matrix M and pixel vector r for the FCLSU method
        Create a matrix delta_M which is δM padded with a row of ones [δM 1.T]
        Creates a vector delta_r [δr 1]
        Output: FCLSU M delta_M: (n_bands, n_end)
                FCLSU r delta_r: (n_bands, 1)
        """
        n_bands, n_end = M.shape
        # Create a matrix delta_M which is δM padded with a row of ones [δM 1.T]
        delta_M = np.ones(shape=(n_bands + 1, n_end))
        delta_M[0:n_bands, :] = delta * M
        #Creates a vector delta_r [δr 1]
        delta_r = np.ones(shape=(n_bands + 1))
        delta_r[0:n_bands] = delta * r

        return delta_M, delta_r


    def UFCLSU(self, err: float = 1e-3, max_endmembers: int = None,  print_output: bool = False) -> tuple[np.array, np.array, np.array]:
        """
        Unsupervised Fully Constrained Least Squares Linear Unmixing.
        Selects desired number of endmembers using a linear predictor model
        This implementation follows MATLAB's lsqnonneg implementation using Langrangian Multipliers

        Output: Abundance Cube a: (nx, ny, n_end)
                Error Cube w: (nx, ny, n_bands)
                Signature Matrix M: (n_bands, n_end)
                Endmember Pixels M_pixels: list of pixel coordinates chosen

        """
        # Collect the endmember pixels
        M_pixels = []

        #Initial Condition: Find the pixel m_0 = argmax(r.T * r), simply the pixel with the greatest inner product
        norm_matrix = np.linalg.norm(self.r, axis=2) # Calculate norm for each pixel across the bands
        max_norm = np.amax(norm_matrix)
        m_0_indices = tuple([idx[0] for idx in np.where(norm_matrix == max_norm)]) # Get the indices of the maximum pixel
        M = self.r[m_0_indices].reshape((-1,1)) # initialize M to be m_0
        M_pixels.append(m_0_indices) # append the index of m_0 to M_pixels

        if print_output:
            print(f"------------|Initialization|------------")
            print(f"Added pixel at {m_0_indices} with inner product: {max_norm}")

        # Create a matrix to store the LSE of the FCLSU unmixing for each pixel
        LSE_matrix = np.ones(shape=(self.nx, self.ny))

        # While any of the LSE is greater than 0, keep iterating
        iter = 1
        delta = 1e-3
        while np.any(LSE_matrix > err) and iter != max_endmembers:
            # Allocating space for delta M and delta r
            delta_M = np.ones(shape=(self.n_bands + 1, M.shape[1]))
            delta_M[0:self.n_bands, :] = delta * M
            delta_r = np.ones(shape=(self.n_bands + 1))

            # calculate the LSE for each pixel in relation to iterations signature matrix M
            for i in range(self.nx):
                for j in range(self.ny):
                    pixel = self.r[i,j]
                    delta_r[0:self.n_bands] = delta * pixel
                    # Calculate the FCLSU abundance weights using the nnls algorithm
                    pixel_abund = self.__nnls(delta_M, delta_r)
                    # Calculate LSE of the FCLSU abundance result
                    LSE_matrix[i,j] = np.sum((np.dot(M, pixel_abund) - self.r[i, j].reshape((-1, 1)))**2)
            
            #Adding the next endmember
            max_lse = np.amax(LSE_matrix)
            m_next_indices = tuple([idx[0] for idx in np.where(LSE_matrix == max_lse)])
            # Adding the pixel with the greatest LSE to the matrix M and adding its indices
            M = np.hstack((M, self.r[m_next_indices].reshape((-1,1))))
            M_pixels.append(m_next_indices) 
            if print_output:
                print(f"------------|Iteration {iter}|------------")
                print(f"Added pixel at {m_next_indices} with LSE: {max_lse}")
            
            iter +=1 
            
        # Calculate the actual FCLSU unmixing
        # Allocates space for abundance and noise cubes
        abundance_cube = np.zeros(shape=(self.nx, self.ny, M.shape[1]))
        noise_cube = np.zeros(shape=(self.nx, self.ny, self.n_bands))

        # Create a matrix delta_M which is δM padded with a row of ones [δM 1.T] (An extra band in the M
        delta_M = np.ones(shape=(self.n_bands + 1, M.shape[1]))
        delta_M[0:self.n_bands, :] = delta * M
        delta_r = np.ones(shape=(self.n_bands + 1))

        # Loop through each pixel
        for i in range(self.r.shape[0]):
            for j in range(self.r.shape[1]):
                # Creates a vector delta_r [δr 1]
                pixel = self.r[i,j]
                delta_r[0:self.n_bands] = delta * pixel
                # Calculates the FCLSU abundance vector and noise vector
                pixel_abundance = self.__nnls(delta_M, delta_r)
                pixel_noise = np.dot(M, pixel_abundance) - self.r[i, j].reshape((-1, 1))
                # Sets the values in the hypercube
                abundance_cube[i, j, :] = pixel_abundance.reshape(M.shape[1])
                noise_cube[i, j, :] = pixel_noise.reshape(self.n_bands)
        return abundance_cube, noise_cube, M, M_pixels
