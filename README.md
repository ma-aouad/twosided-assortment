## Dependencies

Numpy\\
Numba\\
Python 3\\

The code was tested with the versions `numpy==1.19.2` and `numba==0.54.0`.

## Code files and logs

The module `ode_computations.py` contains two methods used to generate the discount functions and to test the approximation guarantees of a given discount function. The corresponding notebook `ode_computations.py` illustrates the two use cases. The discount functions for $\gamma$-nested logit are logged in the file `discount_functions_updated.npy`.

The notebook `model_robustness_experiments.ipynb` contains the methods used to run the numerical experiments (Appendix EC.5) for random instances. Executable commands are provided. The parameters can be changed to reflect the computational settings.
