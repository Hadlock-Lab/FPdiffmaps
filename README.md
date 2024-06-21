# FPdiffmaps
Fokker-Planck diffusion maps for dimensional reduction of stochastic systems. 

General usage is shown in the notebook FP_diffmap_tutorial. Scripts and notebooks for the microglia paper are under the "microglia" subdirectory.

FP tutorial contains the following examples: 
-- Gaussian distribution 

-- Additive Ornstein-Uhlenbeck process

-- Multiplicative Ornstein-Uhlenbeck process with commuting matricies

-- Two independe Birth-Death process

-- 10 D Double well (NOTE: this taks a long time to run)

The microglia directory contains a notebook illustrating the usage and generating the figures contain in the microglia manuscript.



## Package requirements
The basic packages required are 
```
pynndescent, numpy, scipy, matplotlib, scanpy, anndata, pandas
```
Load the scipty functions.py by adding the script's path to the 

```
new_path = "/PATH/TO/functions.py"
if new_path not in sys.path:
    sys.path.append(new_path)
```

## Usage

The general usage for FP diffmaps is the following: 

```
import functions as f
import pynndescent as pynn
import numpy as np

I=pynn.NNDescent(X, n_neighbors=30)
ind,dist=I.query(X,k=15)
nn = f.nn_graph_directed(ind, dist)

nn = nn + nn.T

K = f.getTransitionMatrix(nn)
dMap, eVals = f.getDiffMap(K)

f.plot_diffmnap(dMap)
```

### Microglia

- user must install functions.py at an appropriate place to use properly (site-packages)
- we apologize for any bugs. 
- we tried to explain the steps being performed in each chunk
- this is not meant to be coherent a software package, but instead an illustration of a simple procedure 
- user is encouraged to use jupyter notebooks instead of these scripts

The order in which these should be run are 

1) preprocessing.R
2) run commands in 
3) differential_Expression.py
4) get_crit_trans_field.py
5) classify_clusters.py
6) pseudotime_calc.py
7) Icsmooth_diff_exp.py
8) cibersortX_prep.py
9) cibersortX.results.py

The outputs of each notebook will be used in the next one. It might be wise to rewrite the old data files since they are rather large. The raw data can be obtained synapse with permissions

Olah (raw files only): syn21438358 
Patel (count and raw files): syn28450881
Bulk: syn26207321


