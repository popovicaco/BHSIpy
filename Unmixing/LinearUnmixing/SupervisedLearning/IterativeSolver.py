import numpy as np
np.random.seed(1024)
class LinearUnmixingIterativeSolver:
    """
      Class for Iterative Methods to solve Linear Unmixing Problems
      This Solver does not use convex optimization packages

    Input:  Signature Matrix M: (n_bands, n_end)
            Hyperspectral Image Output r: (nx, ny, n_bands)
    Output: Abundance Cube abundance_cube: (nx, ny, n_end)
            Noise Cube noise_cube: (nx, ny, n_bands)
    """

    def __init__(self, M, r):
        # Getting nx,ny and n_bands
        self.nx, self.ny, self.n_bands = r.shape
        # Getting n_end
        self.n_end = M.shape[1]
        # This is the input signature matrix
        self.M = M
        # This is the desired output hyperspectral image cube
        self.r = r
    
    def run(self, solver_name):
      """
      This is the function specifying which solver the user want to use, it makes running the algorithm easier and more user friendly. 
      """
      if solver_name == "LSU":
        return self.LSU()
      elif solver_name == "CLSU":
        return self.CLSU()
      elif solver_name == "FCLSU":
        return self.FCLSU()

    def LSU(self):
        """
        Unconstrained Least Squares Linear Unmixing
        This implementation calculates the abundance analytically (a = (INV(M.T * M) * M.T)*r)
        Output: Abundance Cube abundance_cube: (nx, ny, n_end)
                Error Cube noise_cube: (nx, ny, n_bands)
        """
        # Reshapes r into a (n_bands, nx*ny) matrix
        r = (self.r.reshape(self.nx * self.ny, self.n_bands)).T
        M = self.M

        # Calculate INV(M.T * M) * M.T
        solution_matrix = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)
        # Calculate our Abundances and Noise
        abundance_matrix = np.dot(solution_matrix, r)
        noise_matrix = np.dot(M, abundance_matrix) - r
        # Reshaping our cubes
        abundance_cube = abundance_matrix.T.reshape((self.nx, self.ny, self.n_end))
        noise_cube = noise_matrix.T.reshape((self.nx, self.ny, self.n_bands))
        return abundance_cube, noise_cube

    @staticmethod
    def __nnls(M, r):
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

    def CLSU(self):
        """
        Constrained Least Squares Linear Unmixing
        This implementation follows MATLAB's lsqnonneg implementation using Langrangian Multipliers
        Output: Abundance Cube a: (nx, ny, n_end)
                Error Cube w: (nx, ny, n_bands)
        """
        # Allocates space for abundance and noise cubes
        abundance_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        noise_cube = np.zeros(shape=(self.nx, self.ny, self.n_bands))
        for i in range(self.r.shape[0]):
            for j in range(self.r.shape[1]):
                pixel_abundance = self.__nnls(self.M, self.r[i, j])
                pixel_noise = np.dot(self.M, pixel_abundance) - self.r[i, j].reshape((-1, 1))
                abundance_cube[i, j, :] = pixel_abundance.reshape(self.n_end)
                noise_cube[i, j, :] = pixel_noise.reshape(self.n_bands)

        return abundance_cube, noise_cube

    def FCLSU(self):
        """
        Fully Constrained Least Squares Linear Unmixing
        This implementation follows MATLAB's lsqnonneg implementation using Langrangian Multipliers
        Output: Abundance Cube a: (nx, ny, n_end)
                Error Cube w: (nx, ny, n_bands)
        """
        delta = 1e-3  # Controls convergence
        # Allocates space for abundance and noise cubes
        abundance_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        noise_cube = np.zeros(shape=(self.nx, self.ny, self.n_bands))

        # Create a matrix delta_M which is δM padded with a row of ones [δM 1.T] (An extra band in the M)
        delta_M = np.ones(shape=(self.n_bands + 1, self.n_end))
        delta_M[0:self.n_bands, :] = delta * self.M
        delta_r = np.ones(shape=(self.n_bands + 1))
        # Loop through each pixel
        for i in range(self.r.shape[0]):
            for j in range(self.r.shape[1]):
                # Creates a vector delta_r [δr 1]
                delta_r[0:self.n_bands] = delta * self.r[i, j]
                # Calculates the FCLSU abundance vector and noise vector
                pixel_abundance = self.__nnls(delta_M, delta_r)
                pixel_noise = np.dot(self.M, pixel_abundance) - self.r[i, j].reshape((-1, 1))
                # Sets the values in the hypercube
                abundance_cube[i, j, :] = pixel_abundance.reshape(self.n_end)
                noise_cube[i, j, :] = pixel_noise.reshape(self.n_bands)
        return abundance_cube, noise_cube
