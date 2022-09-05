import numpy as np
import cvxpy as cp 

class LinearUnmixingAnalyticalSolver:
    """
    Class for LinearUnmixing Methods

    Input:  Signature Matrix M: (num_bands, num_endmembers)
            Hyperspectral Image Output r: (num_pixels_x, num_pixels_y, num_bands)
    Output: Aboundance Vector a: (num_pixels_x, num_pixels_y, num_endmembers)
            Error Vector w: (num_pixels_x, num_pixels_y, num_bands)
    """

    def __init__(self, M, r):
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
      LSU stands for Unconstrained Least Squares LinearUnmixing
      This function derives the abundance vecotr and the error (noise) vector using the standard non-contrained least squares approach. 
      """

      M = self.M
      r = self.r
      coefficient_matrix = np.matmul(np.linalg.inv(np.matmul(M.T,M)),M.T)

      a_final = [] # abundance vector list
      w_final = [] # error list
      for i in range(r.shape[0]):
        for j in range(r.shape[1]):
          a = np.matmul(coefficient_matrix,r[i][j])
          w = np.matmul(M, a)-r[i][j]
          a_final.append(a)
          w_final.append(w)
      return np.array(a_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), np.array(w_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)

    def CLSU(self):
      """
      CLSU stands for Non-Negative Constrained Least Squares LinearUnmixing
      This function derives the abundance vecotr and the error (noise) vector with the non-negative contrained least squares approach. 
      """

      M = self.M
      r = self.r

      a_final = []
      w_final = []
      for i in range(r.shape[0]):
        for j in range(r.shape[1]):
          a = cp.Variable(self.num_endmembers)
          # initialize QP problem
          prob1 = cp.Problem(cp.Minimize((cp.norm(M@a-r[i][j]))**2), [a>=0]) # The first argument is the objective function of the optimization problem, the second argumnet is the list of constraints
          # solve the problem
          sol = prob1.solve(solver=cp.CVXOPT,verbose=False)
          # append solution
          for variable in prob1.variables():
            a_final.append(list(variable.value))
          w = np.matmul(M, np.array(a_final[-1]))
          w_final.append(w)
      return np.array(a_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), np.array(w_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)

    def FCLSU(self):
      """
      FCLSU stands for Fully Constrained Least Squares LinearUnmixing
      This function derives the abundance vecotr with the fully contrained least squares approach. 
      """
      M = self.M
      r = self.r

      a_final = []
      w_final = []
      for i in range(r.shape[0]):
        for j in range(r.shape[1]):
          a = cp.Variable(self.num_endmembers)
          # initialize QP problem
          prob1 = cp.Problem(cp.Minimize((cp.norm(M@a-r[i][j]))**2), [cp.sum(a)==1, a>=0]) # The first argument is the objective function of the optimization problem, the second argumnet is the list of constraints
          # solve the problem
          sol = prob1.solve(solver=cp.CVXOPT,verbose=False)
          # append solution
          for variable in prob1.variables():
            a_final.append(list(variable.value))
          w = np.matmul(M, np.array(a_final[-1]))
          w_final.append(w)
      return np.array(a_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_endmembers), np.array(w_final).reshape(self.num_pixels_x, self.num_pixels_y, self.num_bands)
