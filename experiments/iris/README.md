# Iris Experiments

Each folder contains code from our experiments with the Iris dataset. 

To generate data, simply execute (inside a subfolder)
```
python run.py
```
Note that with ReLUs, not all initializations lead to convergence, so we trained 15 networks (without measuring EI) with `python relu_runs.py`, chose a few runs that converged, then measured the EI of those with `python compute_eis.py`.

Figures can then be made using the `visualize.ipynb` notebooks. 
