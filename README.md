# Timefractional Diffusion

This repository provides a MFEM based implementation to simulate the 
time-fractional heat equation
$$ ((\partial_t)^\alpha  - \nabla \cdot D \nabla) u = 0$$ 
with homogenous Dirichlet BC. We use MFEM for the spatial discretization
and to compute the mass and stiffness matrix. For the time integration, we 
employ the modified Crank-Nicholson algorithm introduce by 
[Khristenko and Wohlmuth (2021)](https://arxiv.org/abs/2102.05139) (Lemma 5.6).
The algorithm is enclosed in a simple for loop.

## Dependencies

The project has a few dependencies:

* MFEM: finite element software needed for spatial discretization. Available via
  their [GitHub](https://github.com/mfem/mfem/)
* CNPY: The project requires the adaptive Antoulas-Anderson (AAA) algorithm 
  which is available for C++ and we had to fall back to Python. The results are 
  saved in the `.npy` format and later read with `CNPY` in the C++ code.
* Python: We use a conda environment to execute the python scripts. The 
  environment can be found in `py_scripts/conda_environment.yml` and be 
  installed with `conda env create -f py_scripts/conda_environment.yml`
* We make use of a library called `mittag_leffler` available via 
  [GitHub](https://github.com/khinsen/mittag-leffler) that should be copied into
  the project directory, i.e. `timefractional-diffusion/mittag_leffler`. We had 
  to remove some `.` from the imports in the library to get the scripts in 
  `py_scripts` running. 
* glVis for visualization of u(t,x) available via [GitHub](). 


## Setting up the conda environment

```bash
conda env create -f py_scripts/conda_environment.yml
conda list | grep fractional
conda activate fractional
```

## Running the python scripts

There are two python scripts `aaa_weights.py` and `visualize_ML.py` that both
should be executed while being in the project directory. The former computes 
coefficients via the AAA algorithm, the latter visualizes the analytic 1D 
solution. For more infos, consult the scripts or their help messages.
```bash
python py_scripts/aaa_weights.py --help
python py_scripts/aaa_weights.py --alpha 0.5 --visualize
python py_scripts/visualize_ML.py --help
python py_scripts/visualize_ML.py --alpha 0.5
```

## Simulating the Timefractional-Heat equation

```bash
mkdir build && cd build
cmake .. && make -j <num_processors>
ctest  # run the few available tests to avoid bad surprises
./tf-diffusion --help # see available parameters
./tf-diffusion -d 1 -r 7 -mi 500 # run 1D simulation 
```

## Visualization

Start glvis in server mode to listen before running `./tf-diffusion`.

```bash
./glvis/build/glvis
```

## Over all workflow

```bash
python py_scripts/aaa_weights.py --alpha 0.5 # compute expansion coefficients
./glvis/build/glvis -mac # start glvis in different terminal
cd build
./tf-diffusion -d 1 -r 7 -mi 500 # solve PDE numerically with coefficients
cd ..
python py_scripts/visualize_ML.py --alpha 0.5 # compare glVis with analytic sol.
```
