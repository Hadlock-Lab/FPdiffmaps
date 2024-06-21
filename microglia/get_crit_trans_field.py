import scanpy as scp
#NOTE: I CHANGED THE DIFFMAP NORMALIZATION TO THE SDE ONE
import anndata as ad
import numpy as np
import pandas as pd
import scipy as spy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import re
import pynndescent as pynn

def First(val):
    return val[0]
def Second(val):
    return val[1]
def Third(val):
    return val[2]
def Fourth(val):
    return val[3]

from sklearn.neighbors import KernelDensity

def getIc(dat):
    num=np.tril(np.corrcoef(dat, rowvar=False),k=-1)
    num=np.abs(num[num>0]).mean()
    den=np.tril(np.corrcoef(dat, rowvar=True),k=-1)
    den=den[den>0].mean()
    return num/den


#THIS IS A SCRIPT TO CALCULATE THE CRITICAL TRANSITION FIELD

data=ad.read_h5ad('./data.h5ad') # use the correct path
"""
genes=[]
for i in range(5):
    for j in range(200):
        g=data.uns['rank_genes_groups']['names'][j][i]
        genes.append(g)
genes=list(set(genes))

dataUse=data[:,genes]
X=np.array(dataUse.X)
"""

X=data.obsm['X_pca']
#X=X[:,:15]

I=pynn.NNDescent(X, n_neighbors=50)
ind,dist=I.query(X,k=30)
IcFin=[]
print('calculating')
for i in range(X.shape[0]):
    x=X[ind[i],:]
    Ic=getIc(x)
    IcFin.append([i,Ic])
print('saving')
finDat=pd.DataFrame(data=IcFin, columns=['index', 'Ic'])
finDat.to_csv('./patelOlahIcValsPCA_all.csv')
print('done')
