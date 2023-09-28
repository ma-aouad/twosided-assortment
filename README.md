## Dependencies

Numpy,
Numba,
Matplotlib,
Python 3

The code was tested with the versions `numpy==1.19.2`, `numba==0.54.0` and `numpy==1.24.3`, `numba==0.57.0`.

## Code files and logs

The module `src/ode_computations.py` contains two methods used to generate the discount functions and to test the approximation guarantees of a given discount function. The notebook `replicate exhibits/main_figures.ipynb` illustrates the use cases and replicate the main figures. The outputs are logged in the folder `outputs/`.

The notebook `model_robustness_experiments.ipynb` contains the methods used to run the numerical experiments (Appendix EC.5) for random instances. Executable commands are provided. The parameters can be changed to reflect the computational settings.