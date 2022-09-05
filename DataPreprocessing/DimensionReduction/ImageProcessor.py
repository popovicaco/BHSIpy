import numpy as np

class LinearDimensionReducer:
    """
    This class is responsible for Dimension Reduction/ Selection methods
    pertaining to Hyperspectral Datasets

    Input:  Hyperspectral Image Output r: (nx, ny, n_bands)
    Output: Reduced Hyperspectral Image (nx, ny, _ )
    """

    def __init__(self, r):
        # Getting nx, ny, and n_bands
        self.nx, self.ny, self.n_bands = r.shape
        # Hyperspectral Cube
        self.r = r
    
    def __choose_initial_bands(self):
        """
        Determines the indices of the two initial dissimilar hyperspectral bands 
        using the relative SAM angle between the two bands. The bands with the 
        highest relative angle are chose as the two initials bands
        Reference: Q. Du and H. Yang, "Similarity-Based Unsupervised Band Selection for 
                   Hyperspectral Image Analysis," in IEEE Geoscience and Remote Sensing 
                   Letters, vol. 5, no. 4, pp. 564-568, Oct. 2008, doi: 10.1109/LGRS.2008.2000619.
        :return  band_1, band_2, relative_angle: indices of the initial bands and the relative angle between them
        """

        next_band_idx, curr_band_idx, prev_band_idx = None, None, None

        # Reshaping the hyperspectral cube into a h_matrix of shape (nx*ny, n_bands)
        h_matrix = np.zeros(shape=(self.nx*self.ny, self.n_bands))
        for i in range(self.n_bands):
            # Reshaping the band into a (nx*ny, 1) vector
            h_matrix[:,i] = self.r[:,:,i].flatten() 

        # Starting the Initial Band Algorithm
        curr_band_idx = 0

        # Create a vector to store SAM values for each band with respoect to the current band
        angle_vec = np.zeros(shape=(1,self.n_bands))
        for i in range(self.n_bands):
            if i != curr_band_idx:
                test_band = h_matrix[:,i]
                curr_band = h_matrix[:,curr_band_idx]
                test_norm = np.linalg.norm(test_band)
                curr_norm = np.linalg.norm(curr_band)
                # Calculate and store the SAM measure between the two vectors
                angle_vec[:,i] = np.arccos(np.dot(test_band, curr_band)/(test_norm*curr_norm))
            else:
                angle_vec[:,i] = 0

        # Set the previous band to the current band and current band to the most dissimilar band
        prev_band_idx = curr_band_idx
        next_band_idx = np.argmax(angle_vec, axis=1)[0]

        # While the previous band and the next band are not equal, iterate
        while next_band_idx != prev_band_idx:
            # Updating the indices
            prev_band_idx = curr_band_idx
            curr_band_idx = next_band_idx
            next_band_idx = 0

            # Create a vector to store SAM values for each band with respoect to the current band
            angle_vec = np.zeros(shape=(1,self.n_bands))
            for i in range(self.n_bands):
                if i != curr_band_idx:
                    test_band = h_matrix[:,i]
                    curr_band = h_matrix[:,curr_band_idx]
                    test_norm = np.linalg.norm(test_band)
                    curr_norm = np.linalg.norm(curr_band)
                    # Calculate and store the SAM measure between the two vectors
                    angle_vec[:,i] = np.arccos(np.dot(test_band, curr_band)/(test_norm*curr_norm))
                else:
                    angle_vec[:,i] = 0

            next_band_idx = np.argmax(angle_vec, axis=1)[0]
        
        # return the index of the initial bands and the relative angle between them
        return curr_band_idx, next_band_idx, angle_vec[:,next_band_idx][0]

    def LinearPredictionSelector(self, num_desired_bands = 2, print_output = False):
        """
        Performs Similar Based LP Band Selection. Starting with Initially Chosen Bands,
        subsequent bands are chosen based on the maximum unconstrained least squares error 
        given by the linear model: a_0 + a_1*BAND_1 + ... + a_m*BAND_M = BAND_PRED

        Reference: Q. Du and H. Yang, "Similarity-Based Unsupervised Band Selection for 
                   Hyperspectral Image Analysis," in IEEE Geoscience and Remote Sensing 
                   Letters, vol. 5, no. 4, pp. 564-568, Oct. 2008, doi: 10.1109/LGRS.2008.2000619.
        :param num_desired_bands: the number of bands to select, default is 2
        :param print_output: print the output of the selections, default is False
        :return  band_selected_r, indices: hyperspectral cube with selected bands and corresponding indices selected
        """
        # Reshaping the hyperspectral cube into a h_matrix of shape (nx*ny, n_bands)
        h_matrix = np.zeros(shape=(self.nx*self.ny, self.n_bands))
        for i in range(self.n_bands):
            # Reshaping the band into a (nx*ny, 1) vector
            h_matrix[:,i] = self.r[:,:,i].flatten() 
        
        # Creating a list to store the selected initial indices and calculate the index of the initial bands B1 and B2
        num_selected_bands = 2
        indices = [0,0]
        indices[0], indices[1], err_initial = self.__choose_initial_bands()

        if print_output:
            print("Selecting " + str(num_desired_bands) + " bands...")
            print("Current Initial Bands: " + str(indices))
            print("Relative Angle between Initial Bands: " + str(err_initial))
            print("------------------")
        
        # Start of the algorithm
        while num_selected_bands != num_desired_bands:
            # Create a matrix X = [1, BAND1, ... BAND_S]  of shape (nx*ny, num_selected_bands+1)
            X = np.ones(shape=(self.nx*self.ny, num_selected_bands+1))
            X[:,1::] = h_matrix[:,indices]

            # Calculating the MP Pseudo Inverse to min||BAND - BAND_PRED||_2
            X_pinv = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)

            # Create a vector to store the linear predictor errors
            lpe_vec = np.zeros(shape=(1,self.n_bands))

            # Calculating the LSE for each band wrt to the linear model: a_0 + a_1*BAND_1 + ... + a_m*BAND_M = BAND_PRED
            for i in range(self.n_bands):
                # Calculating the Unconstrained Predicted Value
                weights = np.dot(X_pinv, h_matrix[:,i])
                band_pred = np.dot(X, weights)
                # Calculating the LSE 
                lpe_vec[:,i] = np.linalg.norm(h_matrix[:,i] - band_pred)
            
            # Append the band with the largest LSE
            indices.append(np.argmax(lpe_vec, axis=1)[0])
            num_selected_bands += 1

            if print_output:
                print("Adding Band " + str(np.argmax(lpe_vec, axis=1)[0]))
                print("Current Selected Bands: " + str(indices))
                print("Current Selected Bands Error: " + str(np.max(lpe_vec, axis=1)[0]))
                print("Number of Selected Bands: " + str(num_selected_bands))
                print("------------------")

        # Sort the indices in ascending order and return the hyperspectral cube with the selected bands
        indices = sorted(indices)
        selected_cube = self.r[:,:,indices]
        return selected_cube, indices

    def PrincipalComponentAnalysis(self, num_principal_components = 2):
        """
        Performs Principal Component Analysis on the Hyperspectral Image. Projects Hyperspectral Image onto the first
        (num_principal_components) eigenvectors of the covariance matrix for each band with the maximum variance/eigenvalue.
        :param num_principal_components: number of principal components, default is 2
        :return  reduced_cube, e_vals, e_vecs: pca-reduced hyperspectal cube and the respective eigenvalues
        """ 
        # Reshaping the hyperspectral cube into a h_matrix of shape (nx*ny, n_bands)
        h_matrix = np.zeros(shape=(self.nx*self.ny, self.n_bands))
        for i in range(self.n_bands):
            # Reshaping the band into a (nx*ny, 1) vector
            h_matrix[:,i] = self.r[:,:,i].flatten() 
        
        for i in range(self.n_bands):
            # Reshaping the band into a (nx*ny, 1) vector
            h_matrix[:,i] = (h_matrix[:,i] - np.mean(h_matrix[:,i]))/np.std(h_matrix[:,i])

        #computes the covariance matrix
        covar_matrix=np.cov(h_matrix.T)

        # Performs eigenvalue/eigenvector decomposition on the covariance matrix
        e_vals, e_vecs = np.linalg.eig(covar_matrix)
        # Sorting the eigenvalues in decreasing order
        indices = np.arange(0,len(e_vals),1)
        indices = [x for _,x in sorted(zip(e_vals,indices))]
        indices = indices[::-1]
        e_vals = e_vals[indices]
        e_vecs = e_vecs[:,indices]

        #extracting the top (num_principal_components) vectors and their eigenvalues
        e_vecs=e_vecs[:,:num_principal_components]
        e_vals=e_vals[:num_principal_components]
        # projecting the bands into the principal components of shape (nx*ny, n_bands)
        reduced_bands = np.dot(e_vecs.T, h_matrix.T).T

        #reshaping the reduced bands into a cube of shape (nx, ny, num_principal_components)
        reduced_cube = reduced_bands.reshape((self.nx, self.ny,num_principal_components))

        return reduced_cube, e_vals, e_vecs

