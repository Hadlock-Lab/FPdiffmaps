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
