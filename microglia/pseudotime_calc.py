import scanpy as scp
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy as spy
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import gget
import pynndescent as pynn
import re
from utils import functions as f
from time import time
from sklearn.cluster import DBSCAN

data=ad.read_h5ad('./data.h5ad')

###################################################
## note that we changed the transition matrix in the anndata frame
## so we have to recalculate it to compute the pseudotime
###################################################

X=data.obsm['X_pca']
I=pynn.NNDescent(X, n_neighbors=30)
ind,dist=I.query(X,k=30)

nnGraph = np.zeros(shape=(X.shape[0],X.shape[0]))
#sig = np.median(dist[:,1:]) ## for additive noise
a = 0
sig = np.mean(dist[:,1:], axis=1)
siga = sig**a
norm = sparse.csr_matrix(np.diag(siga))

for i in range(X.shape[0]):
    nnGraph[ind[i], i] = np.exp( - (dist[i]**2)/(sig[i]**2))
    nnGraph[i,i] = 0

nn = sparse.csr_matrix(nnGraph)
nn = nn @ norm

K= f.getTransitionMatrix(nn)
dMap, eVals = f.getDiffMap(K)

data.obsm['connectivities'] = K


data.uns['iroot'] = 10928
## this choice is somewhat arbitrary.
## one can reasonably choose any point with large Ic in the homeostatic cluster.

ind_root = data.uns['iroot']
scp.tl.dpt(data)
dptSp = data.obs['dpt_pseudotime']

data.write('data_w_ptime.h5ad')

# look at the CTF as a function of pseudotime
for i in range(7):
    ind = data.obs.query(f'leidenRefined == {i}').index.values.astype('int')
    Y = data[ind,:].obs['Ic_all'].values
    X = dptSp[ind]
    
    plt.scatter(X,Y,c=f'C{i}', s=4)
    plt.title(f'cluster {i}')
    plt.xlabel('pseudoradius')
    plt.ylabel('critical transition parameter')
    plt.savefig(f'./pseudotime_Ic_cluster{i}.png')
    plt.show()


## calculate smoothed version of CTF
Z_tot = []
for i in range(7):
    ind = data.obs.query(f'leidenRefined == {i}').index.values
    Z = data[ind,:].obs[['dpt_pseudotime','Ic_all']].values.tolist()
    Z.sort(key = f.First)
    X = [z[0] for z in Z]
    Y = [z[1] for z in Z]
    Y = f.non_uniform_savgol(X,Y,51, 5)
    Z = list(zip(ind,Y))
    Z_tot.append(Z)
    plt.scatter(X,Y,c=f'C{i}', s=4)
    plt.title(f'cluster {i}')
    plt.xlabel('pseudoradius')
    plt.ylabel('critical transition parameter')
    #plt.xlim(0,)
    plt.show()

for z in Z_tot:
    z_arr= np.array(Z)
    data.obs.loc[z_arr[:,0],'IcSm_all'] = z_arr[:,1].astype(float)

