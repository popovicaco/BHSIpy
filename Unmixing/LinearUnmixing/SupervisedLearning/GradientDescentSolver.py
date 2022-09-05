import numpy as np
import time

np.random.seed(10)

class LinearUnmixingGradientDescentSolver:
  """
  Class for Linear Unmixing Methods Using Gradient Descent
  
  Input:  Signature Matrix M: (num_bands, num_endmembers)
          Hyperspectral Image Output r: (num_pixels_x, num_pixels_y, num_bands)
  Output: Aboundance Vector a: (num_pixels_x, num_pixels_y, num_endmembers)
          Error Vector w: (num_pixels_x, num_pixels_y, num_bands)
  """

  def __init__(self, M, r, learning_rate = 1e-3, tolerance = 1e-3):

    # This is the number of pixels along the x axis of the input image
    self.num_pixels_x = r.shape[0]
    # This is the number of pixels along the y axis of the input image
    self.num_pixels_y = r.shape[1]
    # This is the number of bands/wavelength along the z axis of the input image
    self.num_bands = M.shape[0]
    # This is the total number of endmembers being studied
    self.num_endmembers = M.shape[1]
    # This is the input signature matrix
    self.M = M
    # This is the desired output hyperspectral image cube
    self.r = r
    # This is the total number of pixels
    self.num_pixels = self.num_pixels_x * self.num_pixels_y
    # This is the Learning Rate
    self.learning_rate = learning_rate
    # This is the tolerance for gradient descent method
    self.tolerance = tolerance

  def run(self, solver_name):
    """
    This is the function specifying which solver the user want to use, it makes running the algorithm easier and more user friendly. 
    """
    if solver_name == "LSU":
      return self.LSUGD()
    elif solver_name == "CLSU":
      return self.CLSUGD()
    elif solver_name == "FCLSU":
      return self.FCLSUGD()
  
  def LSUGD(self):
    """
    LSUGP stands for Unconstrained Least Squares Linear Unmixing with Gradient Descent
    This function derives the abundance vecotr with the Gradient Descent approach. 
    """

    M = self.M
    r = (self.r.reshape(self.num_pixels_x*self.num_pixels_y, self.num_bands)).T

    # defining the Loss/Objective Function
    def loss(abundance, M, r):
      return np.linalg.norm(np.dot(M,abundance) - r)**2
    
    # initializing a random abundance matrix
    previous_abundance = np.random.uniform(size=(self.num_endmembers, self.num_pixels_x*self.num_pixels_y))
    current_abundance = np.zeros((self.num_endmembers, self.num_pixels_x*self.num_pixels_y))
    temp = None

    num_iterations = 0
    start_time = time.perf_counter()
    print("START TRAINING")
    while (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) >= self.tolerance) and (num_iterations != 1e5):
      num_iterations += 1
      # compute the gradient
      temp = current_abundance
      
      gradient = 2 * np.dot(M.T, np.dot(M, previous_abundance) - r)
      current_abundance = previous_abundance - self.learning_rate*gradient
      previous_abundance = temp
      if (num_iterations == 1e5) and (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) > self.tolerance):
        print("Maximum iterations has passed, but no solutions are found")
      if (num_iterations) % 100 == 0:
        print("Epoch: ", str(num_iterations), "Loss: ", str(loss(current_abundance, M, r)))

    end_time = time.perf_counter()
    current_abundance = np.nan_to_num(current_abundance, copy=True, nan=0.0, posinf=1, neginf=0)
    print("Solution is found after " + str(num_iterations) + " epochs", ", with the final Loss equal to: " + str(loss(current_abundance, M, r)))
    a_final = current_abundance.T
    w_final = np.dot(M, current_abundance) - r
    print(f"Complete the LSU Gradient Descent Algorithm in {end_time - start_time:0.2f} seconds")
    return a_final.reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), (w_final.T).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)

  def CLSUGD(self):
    """
    CLSUGP stands for Non-Negative Constrained Least Squares Linear Unmixing with Gradient Descent
    This function derives the abundance vecotr with the Gradient Descent approach. 
    """

    M = self.M
    r = (self.r.reshape(self.num_pixels_x*self.num_pixels_y, self.num_bands)).T

    # defining the Loss/Objective Function
    def loss(abundance, M, r):
      return np.linalg.norm(np.dot(M,abundance) - r)**2

    # initializing a random abundance matrix
    previous_abundance = np.random.uniform(size=(self.num_endmembers, self.num_pixels_x*self.num_pixels_y))
    current_abundance = np.random.uniform(size=(self.num_endmembers, self.num_pixels_x*self.num_pixels_y))

    # Initializing the learning rate matrix
    learning_rate = np.ones((self.num_endmembers, self.num_pixels_x*self.num_pixels_y))*self.learning_rate

    num_iterations = 0
    start_time = time.perf_counter()
    print("START TRAINING")
    while (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) >= self.tolerance)  and (num_iterations != 1e5):
      num_iterations += 1
      # computing the gradient
      previous_abundance = current_abundance
      gradient = 2 * np.dot(M.T, np.dot(M, previous_abundance) - r)
      
      learning_rate_unchanged = (gradient < 0)*learning_rate
      # Compute the lower & upper bound of the learning rate changed, those whose gradient are larger than 0
      learning_rate_lower_bound = 0
      learning_rate_upper_bound = 1/gradient
      learning_rate_upper_bound = np.nan_to_num(learning_rate_upper_bound, copy=True, nan=0.0, posinf=1, neginf=0)

      learning_rate_changed = np.random.uniform(learning_rate_lower_bound, learning_rate_upper_bound) * (gradient > 0)
      learning_rate_changed = np.clip(learning_rate_changed, 0, self.learning_rate)
      
      # Compute the total learning rate
      learning_rate = learning_rate_unchanged + learning_rate_changed

      # Gradient Descent
      current_abundance = previous_abundance - learning_rate * previous_abundance * gradient
      if (num_iterations == 1e5) and (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) > self.tolerance):
        print("Maximum iterations has passed, but no solutions are found")
      if (num_iterations) % 100 == 0:
        print("Epoch: ", str(num_iterations), "Loss: ", str(loss(current_abundance, M, r)))

    end_time = time.perf_counter()
    current_abundance_for_loss = np.nan_to_num(current_abundance, copy=True, nan=0.0, posinf=1, neginf=0)
    print("Solution is found after " + str(num_iterations) + " epochs", ", with the final Loss equal to: " + str(loss(current_abundance_for_loss, M, r)))
    a_final = current_abundance.T
    w_final = np.dot(M, current_abundance) - r
    print(f"Complete the CLSU Gradient Descent Algorithm in {end_time - start_time:0.2f} seconds")
    return a_final.reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), (w_final.T).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)

  def FCLSUGD(self):
    """
    FCLSUGP stands for Fully Constrained Least Squares Linear Unmixing with Gradient Descent
    This function derives the abundance vecotr with the Gradient Descent approach. 
    """

    M = self.M
    r = (self.r.reshape(self.num_pixels_x*self.num_pixels_y, self.num_bands)).T

    # defining the Loss/Objective Function
    def loss(abundance, M, r):
      return np.linalg.norm(np.dot(M,abundance) - r)**2

    # initializing a random abundance matrix
    previous_abundance = np.random.uniform(size=(self.num_endmembers, self.num_pixels_x*self.num_pixels_y))
    current_abundance = np.random.uniform(size=(self.num_endmembers, self.num_pixels_x*self.num_pixels_y))
    previous_abundance = previous_abundance/np.sum(previous_abundance, 0)

    # Initializing the learning rate matrix
    learning_rate = np.ones((self.num_endmembers, self.num_pixels_x*self.num_pixels_y))*self.learning_rate

    num_iterations = 0
    start_time = time.perf_counter()
    print("START TRAINING")
    while (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) >= self.tolerance) and (num_iterations != 1e5):
      num_iterations += 1

      # computing the gradient
      previous_abundance = current_abundance
      gradient = 2 * np.dot(M.T, np.dot(M, previous_abundance) - r)
      
      learning_rate_unchanged = (gradient < 0)*learning_rate
      # Compute the lower & upper bound of the learning rate changed, those whose gradient are larger than 0
      learning_rate_lower_bound = 0
      learning_rate_upper_bound = 1/(gradient - gradient * previous_abundance)
      learning_rate_upper_bound = np.nan_to_num(learning_rate_upper_bound, copy=True, nan=0.0, posinf=1, neginf=0)

      learning_rate_changed = np.random.uniform(learning_rate_lower_bound, learning_rate_upper_bound) * (gradient > 0)
      learning_rate_changed = np.clip(learning_rate_changed, 1e-3, self.learning_rate)
      
      # Compute the total learning rate
      learning_rate = learning_rate_unchanged + learning_rate_changed

      # Gradient Descent
      current_abundance = previous_abundance - learning_rate * previous_abundance * gradient - learning_rate * previous_abundance * gradient * previous_abundance
      if (num_iterations == 1e5) and (abs(loss(current_abundance, M, r) - loss(previous_abundance, M, r)) > self.tolerance):
        print("Maximum iterations has passed, but no solutions are found")
      if (num_iterations) % 100 == 0:
        current_abundance_for_loss = np.nan_to_num(current_abundance, copy=True, nan=0.0, posinf=1, neginf=0)
        print("Epoch: ", str(num_iterations), "Loss: ", str(loss(current_abundance_for_loss, M, r)))
    
    end_time = time.perf_counter()
    current_abundance_for_loss = np.nan_to_num(current_abundance, copy=True, nan=0.0, posinf=1, neginf=0)
    print("Solution is found after " + str(num_iterations) + " epochs", ", with the final Loss equal to: " + str(loss(current_abundance_for_loss, M, r)))
    a_final = current_abundance.T
    w_final = np.dot(M, current_abundance) - r
    print(f"Complete the FCLSU Gradient Descent Algorithm in {end_time - start_time:0.2f} seconds")
    return a_final.reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), (w_final.T).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)
