import numpy as np


class AnalyticalSegmentationSolver:
    """
    Class for Supervised Hyperspectral Segmentation

    Input:  Signature Matrix M: (n_bands, n_end)
            Hyperspectral Image Output r: (nx, ny, n_bands)
    Output: Segmented Hyperspectral Image measure_cube: (nx, ny, n_end)
            Relative Spectral Discriminatory Entropy rdse_matrix: (nx,ny)
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
        # This is the total number of pixels
        self.num_pixels = self.nx * self.ny

    @staticmethod
    def __calc_sam(reference_spectra, pixel_spectra) -> float:
        """
        Calculates the Spectral Angle Mapper measure, the similarity of test pixel spectra to known endmember spectra
        This similarity in radians is calculated as the relative angle between the two vectors in n-dimensional space
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :return sam_measure: Spectral Angle Mapper Measure
        """
        # Calculates the norm of both vectors
        reference_spectra_norm = np.linalg.norm(reference_spectra)
        pixel_spectra_norm = np.linalg.norm(pixel_spectra)
        # Calculates the cosine distance between both vectors
        cosine_distance = np.dot(pixel_spectra, reference_spectra) / (reference_spectra_norm * pixel_spectra_norm)
        # Calculates the SAM measure between both vectors
        sam_measure = np.arccos(cosine_distance)
        return sam_measure

    @staticmethod
    def __calc_sid(reference_spectra, pixel_spectra) -> float:
        """
        Calculates the Spectral Information Divergence measure, the similarity of test pixel to known endmembers.
        The spectral signatures are viewed as probability distributions and using information theory their probabilistic
        distance is calculated for each band. Then the distance is averages with respect to each probability distribution
        and the value is symmetric.
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :return sid_measure: Spectral Information Discrimination Measure
        """
        # calculates the discrete probability distributions for the pixel and the reference spectras
        p = (pixel_spectra / np.sum(pixel_spectra)) + np.spacing(1)  # avoid division by zero
        q = (reference_spectra / np.sum(reference_spectra)) + np.spacing(1)  # avoid division by zero
        # Calculate the relative cross entropy of q with respect to p and vice versa in nats
        relative_entropy_pq = np.dot(p, np.log(np.divide(p, q)))
        relative_entropy_qp = np.dot(q, np.log(np.divide(q, p)))
        # calculate the spectral information divergence
        sid_measure = relative_entropy_pq + relative_entropy_qp
        return sid_measure

    @staticmethod
    def __calc_sidsin(reference_spectra, pixel_spectra) -> float:
        """
        Calculates the SID(sin) measure, the similarity of test pixel to known endmembers.
        Combines sid and sam using sin
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :return sidsin_measure: SID-SAM sin Measure
        """
        sam = AnalyticalSegmentationSolver.__calc_sam(reference_spectra, pixel_spectra)
        sid = AnalyticalSegmentationSolver.__calc_sid(reference_spectra, pixel_spectra)
        sidsin_measure = sid * np.sin(sam)
        return sidsin_measure

    @staticmethod
    def __calc_sidtan(reference_spectra, pixel_spectra) -> float:
        """
        Calculates the SID(tan) measure, the similarity of test pixel to known endmembers.
        Combines sid and sam using tan
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :return sidtan_measure: SID-SAM tan Measure
        """
        sam = AnalyticalSegmentationSolver.__calc_sam(reference_spectra, pixel_spectra)
        sid = AnalyticalSegmentationSolver.__calc_sid(reference_spectra, pixel_spectra)
        sidtan_measure = sid * np.tan(sam)
        return sidtan_measure

    @staticmethod
    def __calc_entropy(spectral_measure) -> float:
        """
        Calculates the Spectral Discriminatory Entropy of Spectral Measure wrt a reference and test spectra
        Defined as the relative entropy within a normalized measure vector in base 2
        :return pixel_sde: The entropy across a pixel's endmember spectral measures
        """
        # Calculates the spectral discriminatory probability
        spectral_prob = spectral_measure / (np.sum(spectral_measure) + np.spacing(1))  # avoid division by zero
        # SDE = -Σp*log(p)
        pixel_sde = -np.dot(spectral_prob, np.log2(spectral_prob))
        return pixel_sde

    @staticmethod
    def __calc_discrim_pwr(spectral_measure1, spectral_measure2) -> float:
        """
        Calculates the Spectral Discriminatory Power of Spectral Measure wrt a reference and 2 test spectra
        Defined as the max of the ratios of the measure of the reference and one of the test spectra to the other
        :return pixel_dp: The discriminatory power between two spectral signatures with respect to a refrence specral signature
        """
        # Calculates the spectral discriminatory power
        ratio1 = spectral_measure1 / (spectral_measure2 + np.spacing(1))  # avoid division by zero
        ratio2 = spectral_measure2 / (spectral_measure1 + np.spacing(1))  # avoid division by zero
        pixel_dp = max(ratio1, ratio2)
        return pixel_dp

    @staticmethod
    def __calc_jmsam(reference_spectra, pixel_spectra, option="SIN") -> float:
        """
        Calculates the Jeffries Matusita and Spectral Angle Mapper Mixed Measure. 
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :param option: apply SIN/TAN to the SAM angle, default is SIN
        :return jmsam_measure: JMSAM
        """
        # calculate the discrete probability distribution vectors p and q
        p = (pixel_spectra / np.sum(pixel_spectra)) + np.spacing(1)
        q = (reference_spectra / np.sum(reference_spectra)) + np.spacing(1)

        # calculates the JM distance between p and q
        jmd = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

        # calculates SIN(SAM) or TAN(SAM)
        if option == "SIN":
            sam = np.sin(AnalyticalSegmentationSolver.__calc_sam(reference_spectra, pixel_spectra))
        elif option == "TAN":
            sam = np.tan(AnalyticalSegmentationSolver.__calc_sam(reference_spectra, pixel_spectra))
        return jmd * sam

    @staticmethod
    def __calc_ns3(reference_spectra, pixel_spectra, a_min, a_max) -> float:
        """
        Calculates the Normalized spectral similarity score. This is a mixed measure between SAM and spectral amplitude
        difference. The COS one is from a paper.
        :param reference_spectra: a (n_bands,) vector corresponding to the reference endmember spectra
        :param pixel_spectra: a (n_bands,) vector corresponding to the test spectra
        :param a_min: minimum spectral amplitude in image
        :param a_max: maximum spectral amplitude in image
        :return NS3_measure: NS3
        """
        # Get the number of bands N
        n = np.shape(reference_spectra)[0]
        # calculate the spectral amplitude difference A
        a = np.sqrt(np.sum((pixel_spectra - reference_spectra) ** 2) / n)
        # calculate normilized spectral amplituded difference
        a_norm = (a - a_min) / (a_max - a_min)
        # calculates 1-cos(SAM)
        sam = 1 - np.cos(AnalyticalSegmentationSolver.__calc_sam(reference_spectra, pixel_spectra))
        # calculates Ns3 score
        NS3_measure = np.sqrt((a_norm ** 2) + sam ** 2)
        return NS3_measure

    def SpectralAngleMapper(self):
        """
        Spectral Angle Mapper Classifier classifies a hyperspectral cube according to the SAM measure
        Output: Segmented Hyperspectral Image spectral_angle_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        spectral_angle_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        sde_matrix = np.zeros(shape=(self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the test pixel spectra across all the bands
                pixel_spectra = self.r[i, j, :]
                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    spectral_angle_cube[i, j, k] = self.__calc_sam(endmember_spectra, pixel_spectra)
                # calculates the RDSE for the pixel's SAM measure
                sde_matrix[i, j] = self.__calc_entropy(spectral_angle_cube[i, j, :])
        return spectral_angle_cube, sde_matrix

    def SpectralInformationDivergence(self):
        """
        Spectral Information Divergence Classifier classifies a hyperspectral cube according to the SID measure
        Output: Spectral Divergence Measure Cube spectral_divergence_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        spectral_divergence_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        sde_matrix = np.zeros(shape=(self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the test pixel spectra across all the bands
                pixel_spectra = self.r[i, j, :]

                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    # Calculating Inserting the measure value into the cube
                    spectral_divergence_cube[i, j, k] = self.__calc_sid(endmember_spectra, pixel_spectra)

                # calculates the RDSE for the pixel's SID measure
                sde_matrix[i, j] = self.__calc_entropy(spectral_divergence_cube[i, j, :])
        return spectral_divergence_cube, sde_matrix

    def SpectralMixedMeasure(self, option="SIN"):
        """
        The SID - SAM Mixed Measure combines both Spectral Angle Mapper and Spectral Information Divergence into a new
        measure. This measure has two versions for a pixel spectra r and an endmember reference spectra s:
                SID(SIN) = SID(r,s)*sin(SAM(r,s))  and SID(TAN) = SID(r,s)*tan(SAM(r,s))
        Input: option: "SIN" or "TAN"
        Output: Spectral Mixed Measure Cube spectral_mixed_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        spectral_mixed_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        sde_matrix = np.zeros(shape=(self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the test pixel spectra across all the bands
                pixel_spectra = self.r[i, j, :]

                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    # Calculating the SAM and SID measures
                    sam_measure = self.__calc_sam(endmember_spectra, pixel_spectra)
                    sid_measure = self.__calc_sid(endmember_spectra, pixel_spectra)
                    # Calculating the SID-SAM mixed measure according to user preference
                    mixed_measure = 0
                    if option == "SIN":
                        mixed_measure = sid_measure * np.sin(sam_measure)
                    elif option == "TAN":
                        mixed_measure = sid_measure * np.tan(sam_measure)
                    # Inserting the mixed_measure value into the cube
                    spectral_mixed_cube[i, j, k] = mixed_measure
                # calculates the RDSE for the pixel's SIDSAM measure
                sde_matrix[i, j] = self.__calc_entropy(spectral_mixed_cube[i, j, :])
        # Finding the endmember with the lowest SAM value across n_end axis
        return spectral_mixed_cube, sde_matrix

    def JMSAMMixedMeasure(self, option="SIN"):
        """
        Jeffries Matusita and Spectral Angle Mapper classifier classifies a hyperspectral cube according to the JMSAM
        Output: Spectral Divergence Measure Cube spectral_divergence_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        jmsam_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        sde_matrix = np.zeros(shape=(self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the test pixel spectra across all the bands
                pixel_spectra = self.r[i, j, :]
                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    # Calculating and inserting the measure value into the cube
                    if option == "SIN":
                        jmsam_cube[i, j, k] = self.__calc_jmsam(endmember_spectra, pixel_spectra)
                    elif option == "TAN":
                        jmsam_cube[i, j, k] = self.__calc_jmsam(endmember_spectra, pixel_spectra, option="TAN")
                # calculates the RDSE for the pixel's SID measure
                sde_matrix[i, j] = self.__calc_entropy(jmsam_cube[i, j, :])
        return jmsam_cube, sde_matrix

    def NS3(self):
        """
        NS3 classifier classifies a hyperspectral cube according to the normalized spectral similarity score
        Output: NS3 Cube spectral_divergence_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        ns3_cube = np.zeros(shape=(self.nx, self.ny, self.n_end))
        sde_matrix = np.zeros(shape=(self.nx, self.ny))
        # calculate ns3 score for every pixel
        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the test pixel spectra across all the bands
                pixel_spectra = self.r[i, j, :]
                # determine minimum and maximum spectral ampilitude difference for each pixel
                a = np.zeros(self.n_end)
                index = 0
                # Get the number of bands N
                n = self.n_bands
                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    a[index] = np.sqrt(np.sum((pixel_spectra - endmember_spectra) ** 2) / n)
                    index += 1
                a_min = np.min(a)
                a_max = np.max(a)
                # Looping through each endmember reference spectra in M
                for k in range(self.n_end):
                    # Grabbing the reference pixel spectra
                    endmember_spectra = self.M[:, k]
                    # Calculating and inserting the measure value into the cube
                    ns3_cube[i, j, k] = self.__calc_ns3(endmember_spectra, pixel_spectra, a_min, a_max)
                # calculates the RDSE for the pixel's SID measure
                sde_matrix[i, j] = self.__calc_entropy(ns3_cube[i, j, :])
        return ns3_cube, sde_matrix

    def DiscriminatoryPower(self, test1, test2, reference, measure):
        """
        The Discriminatory power determins the ability of a measure to distinguish between two sprectra with
        respect to a third. This measure is the max between the ratios of the measure of test1 and reference vs
        the measure of test2 and reference.

        Input:  test1: test spectral signature one
                test2: test spectral signature two
                reference: reference spectral signature
                measure: "SAM", "SID", "SID(SIN)", "SID(TAN)", "JMSAM(SIN)", "JMSAM(TAN)", "NS3"
        Output: Spectral Mixed Measure Cube spectral_mixed_cube: (nx, ny, n_end)
                Spectral Discriminatory Entropy sde_matrix: (nx,ny)
        """
        # initializing spectral meausres
        spectral_measure1 = 0
        spectral_measure2 = 0
        # calculating spectral measures based on measure input
        if measure == "SAM":
            spectral_measure1 = self.__calc_sam(reference, test1)
            spectral_measure2 = self.__calc_sam(reference, test2)
        elif measure == "SID":
            spectral_measure1 = self.__calc_sid(reference, test1)
            spectral_measure2 = self.__calc_sid(reference, test2)
        elif measure == "SID(SIN)":
            spectral_measure1 = self.__calc_sidsin(reference, test1)
            spectral_measure2 = self.__calc_sidsin(reference, test2)
        elif measure == "SID(TAN)":
            spectral_measure1 = self.__calc_sidtan(reference, test1)
            spectral_measure2 = self.__calc_sidtan(reference, test2)
        elif measure == "JMSAM(SIN)":
            spectral_measure1 = self.__calc_jmsam(reference, test1)
            spectral_measure2 = self.__calc_jmsam(reference, test2)
        elif measure == "JMSAM(TAN)":
            spectral_measure1 = self.__calc_jmsam(reference, test1, option="TAN")
            spectral_measure2 = self.__calc_jmsam(reference, test2, option="TAN")
        elif measure == "NS3":
            # determine minimum and maximum spectral ampilitude difference for test 1 and test 2
            a1 = np.zeros(self.n_end)
            a2 = np.zeros(self.n_end)
            index = 0
            # Get the number of bands N
            n = self.n_bands
            # Looping through each endmember reference spectra in M
            for k in range(self.n_end):
                # Grabbing the reference pixel spectra
                endmember_spectra = self.M[:, k]
                a1[index] = np.sqrt(np.sum((test1 - endmember_spectra) ** 2) / n)
                a2[index] = np.sqrt(np.sum((test2 - endmember_spectra) ** 2) / n)
                index += 1
            a1_min = np.min(a1)
            a1_max = np.max(a1)
            a2_min = np.min(a2)
            a2_max = np.max(a2)

            spectral_measure1 = self.__calc_ns3(reference, test1, a1_min, a1_max)
            spectral_measure2 = self.__calc_ns3(reference, test2, a2_min, a2_max)
        # calculate discriminaory power
        dp = self.__calc_discrim_pwr(spectral_measure1, spectral_measure2)
        return dp

    def SegDifference(self, segimage, groundtruth):
        """
        The Segmentation difference calculates the number of pixels that have different classifications
        (not including null classified pixels) divided by the number of pixels that have ground truth data
        for each endmember. This can be used to tell how accurate the segmentation algorithm was. Higher values mean
        less accurate segmentation for that endmember.

        Input:  segimage: segmented image
                groundtruth: ground truth data
        Output: Segmentation Difference Array: An array that contains the difference in the image for each endmember
        """
        segdif = np.zeros((self.n_end)+1)
        totalpixles = np.zeros((self.n_end)+1)
        for i in range(self.nx):
            for j in range(self.ny):
                # Grabbing the pixel endmember classification
                endmember = groundtruth[i, j]
                # looping through each endmenber classification
                for k in range(self.n_end + 1):
                    if k == endmember:
                        # counting number of pixels in ground truth data with that endmember classification
                        totalpixles[k] = totalpixles[k] + 1
                        if segimage[i, j] != endmember:
                            # counting number of pixels in segmented image with different classification than ground truth
                            segdif[k] = segdif[k] + 1
        return segdif / totalpixles
