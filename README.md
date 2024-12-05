# FPdiffmaps
Fokker-Planck diffusion maps for dimensional reduction of stochastic systems. 

General usage is shown in the notebook FP_diffmap_tutorial. Scripts and notebooks for the microglia paper are under the "microglia" subdirectory.

FP tutorial contains the following examples: 

-- Gaussian distribution 

-- Additive Ornstein-Uhlenbeck process

-- Multiplicative Ornstein-Uhlenbeck process with commuting matricies

-- Two independent Birth-Death processes

-- 10 D Double well (NOTE: this taks a long time to run)

The microglia directory contains a notebook illustrating the usage and generating the figures contain in the microglia manuscript.



## Package requirements
The basic packages required are 
```
pynndescent, numpy, scipy, matplotlib, scanpy, anndata, pandas
```
Load the scipt functions.py by adding the script's path to the 

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

NOTE: this is not meant to be a fully modular software package! i believe the unique code is too simple to warrant an entire software suite, but i do provide the essential functions needed for all calculations in the script functions.py. i am not a software engineer and so this code is not necessarily optimized or efficient. i have tried to explain what each chunk in each script is doing so as to help the reader. thank you for understanding and please reach out with any questions.


- user must install functions.py at an appropriate place to use properly 
- we apologize for any bugs. 
- we tried to explain the steps being performed in each chunk
- user is encouraged to use jupyter notebooks instead of these scripts

The order in which these should be run are 

1) preprocessing.R : preprocess and integrate the data
2) diffmap_and_cluster.py : computes diffusion maps, clusters, and pseudobulk analysis. 
4) get_crit_trans_field.py : calculates local critical transition field with a median convolutional filter
5) classification_and_signatures.py : trains an XGBoost model, runs SHAP, and find high-variance SHAP features
6) pseudotime_calc.py : calculates pseudo-redial ordering using diffusion pseudotime
7) Icsmooth_diff_exp.py : smooth Ic vs. pseudoradius by a non-uniform Savitzky-Golay filter, and perform differential expression on top and bottom 5% of cells by Ic value 
9) cibersortX.results.py : calculates KS-test for deconvolution results

The outputs of each notebook will be used in the next one. It might be wise to rewrite the old data files since they are rather large. The raw data can be obtained synapse with permissions

Olah (raw files only): syn21438358 
Patel (count and raw files): syn28450881
Bulk: syn26207321

