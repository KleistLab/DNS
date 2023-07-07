# DNS
Drosophila Neural Superposition

The code in this repository implements the model presented in "Axonal self-sorting without target guidance in Drosophila visual map formation" by Agi, Reifenstein, et al., 2023.

Files:
- randomwalk_growthcone_density.py implements the model, produces movies of simulated growing axons, and performs parameter sweeps of the model
- create_grid_for_all_timesteps.py calculates and stores the density landscapes used by randomwalk_growthcone_density.py
- estimate_average_grid.py fits the two grid-spanning vectors to the experimental data and stores them in a file
- six_receptors_one_movie.py simulates the axons and creates the movie for all six-receptor subtypes simulateously, or a even a full set of all axons if desired
- randomwalk_growthcone_density_noLcell.py is similar to randomwalk_growthcone_density.py but uses the data of the L-cell ablation experiment 
- create_grid_for_all_timesteps_noLcell.py is similar to create_grid_for_all_timesteps.py but uses the data of the L-cell ablation experiment
- estimate_average_grid_noLcell.py is similar to create_grid_noLcell.py but uses the data of the L-cell ablation experiment
- helper_functions.py contains a bunch of general functions being reused in different files
- experimental_files.zip contains the experimental data needed to run the scripts (locations for fronts, heels, front filopodia, L cells)
  
The files *_noLcell.py could have been incorporated into their corresponding wild-type counterparts but we felt that these had already very many branches.
We think that merging these files would have made the code much less readable.


Usage:

  (1) run estimate_average_grid.py, it stores a file containing the fitted grid-spanning vectors
  
  (2) run create_grid_for_all_timesteps.py, it uses this file and creates the density landscape for all time steps of the simulation, stored in .npy files
  
  (3) run randomwalk_growthcone_density.py, it uses the .npy files from (2) and simulates the growing axons 

Use the same order for the *_noLcell.py files.
six_receptors_one_movie.py also requires the steps (1) and (2) first.

Required modules:
numpy, scipy, pandas, matplotlib, numexpr, joblib, json, base64, time, descartes, shapely

The randomwalk_growthcone_density* files use mencoder to combine the created images into a movie.

Make sure to adapt the paths in the code for writing and storing files to your system. 
