## I. Sample Usage of the Analytical Solver

```python 
import scipy.io
# Load Dataset
hyperspectral_unmixing_data = scipy.io.loadmat('hyperspectral_image_sample_dataset.mat')
# Get Signature Matrix M
signature_matrix = hyperspectral_unmixing_data["S0"]
# Get the desired output Hyperspectral Image Cube r
hyperspectral_cube = hyperspectral_unmixing_data["data"]
# Define the Analytical Solver Class
Analytical_Solver = LinearUnmixingAnalyticalSolver(M = signature_matrix, r = hyperspectral_cube)
# Use the Unconstrained Least Squares Linear Unmixing solver to solve for abundance and noise of each endmembers in each pixel
LSU_abundance_result, LSU_noise_result = Analytical_Solver.run("LSU")
# Use the Non-Negative Constrained Least Squares Linear Unmixing solver to solve for abundance and noise of each endmembers in each pixel
CLSU_abundance_result, CLSU_noise_result = Analytical_Solver.run("CLSU")
# Use the Fully Constrained Least Squares Linear Unmixing solver to solve for abundance and noise of each endmembers in each pixel
FCLSU_abundance_result, FCLSU_noise_result = Analytical_Solver.run("FCLSU")
```

## II. Sample Usage of the Gradient Descent Solver

```python 
import scipy.io
# Load Dataset
hyperspectral_unmixing_data = scipy.io.loadmat('hyperspectral_image_sample_dataset.mat')
# Get Signature Matrix M
signature_matrix = hyperspectral_unmixing_data["S0"]
# Get the desired output Hyperspectral Image Cube r
hyperspectral_cube = hyperspectral_unmixing_data["data"]
# Define the Gradient Descent Solver Class
Gradient_Descent_Solver = LinearUnmixingGradientDescentSolver(M = signature_matrix, r = hyperspectral_cube, learning_rate = 5e-3, tolerance = 5e-3)
# Use the Unconstrained Least Squares Linear Unmixing with Gradient Descent solver to solve for abundance and noise of each endmembers in each pixel
LSU_abundance_result, LSU_noise_result = Gradient_Descent_Solver.run("LSU")
# Use the Non-Negative Constrained Least Squares Linear Unmixing with Gradient Descent solver to solve for abundance and noise of each endmembers in each pixel
CLSU_abundance_result, CLSU_noise_result = Gradient_Descent_Solver.run("CLSU")
# Use the Fully Constrained Least Squares Linear Unmixing with Gradient Descent solver to solve for abundance and noise of each endmembers in each pixel
FCLSU_abundance_result, FCLSU_noise_result = Gradient_Descent_Solver.run("FCLSU")
```
## III. Sample Usage of the Iterative Solver

```python 
import scipy.io
# Load Dataset
hyperspectral_unmixing_data = scipy.io.loadmat('hyperspectral_image_sample_dataset.mat')
# Get Signature Matrix M
signature_matrix = hyperspectral_unmixing_data["S0"]
# Get the desired output Hyperspectral Image Cube r
hyperspectral_cube = hyperspectral_unmixing_data["data"]
# Define the Iterative Solver Class
Iterative_Solver = LinearUnmixingIterativeSolver(M = signature_matrix, r = hyperspectral_cube)
# Use the Unconstrained Least Squares Linear Unmixing with Iterative solver to solve for abundance and noise of each endmembers in each pixel
LSU_abundance_result, LSU_noise_result = Iterative_Solver.run("LSU")
# Use the Non-Negative Constrained Least Squares Linear Unmixing with Iterative solver to solve for abundance and noise of each endmembers in each pixel
CLSU_abundance_result, CLSU_noise_result = Iterative_Solver.run("CLSU")
# Use the Fully Constrained Least Squares Linear Unmixing with Iterative solver to solve for abundance and noise of each endmembers in each pixel
FCLSU_abundance_result, FCLSU_noise_result = Iterative_Solver.run("FCLSU")
```

