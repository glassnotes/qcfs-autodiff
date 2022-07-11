# qcfs-autodiff

This repository contains all the scripts and results for the quantum computing fidelity susceptibility with automatic differentiation project. 

 - Required packages can be installed using the provided `requirements.txt` file. (For results in the paper, Python 3.10.4 was used)
 - Data used to produce plots in the paper is provided in the `results-*` directories, along with Jupyter notebooks to generate those plots.

The main script in this directory is `mit_experiments.py`. It can be run using the following command:

```
python mit_experiments.py <L> <n_trials> <folding_fn> <scale_factors>
```

For example, to run the 4-spin case with 50 trials, and error mitigation with CNOT folding and scale factors [1, 3, 5], run: 

```
python mit_experiments.py 4 50 cnot 135
```

Due to randomness in the multiple frameworks/libraries used, running these scripts again will produce data similar to, but not exactly equal to what is provided in the results directories.
