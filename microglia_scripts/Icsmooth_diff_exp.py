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

counts = scp.read_h5ad('./counts.h5ad')
data = scp.read_h5ad('.data_w_ptime.h5ad')
data.obs.set_index('index',drop = False, inplace = True)

####################################################
## perform differential expression based on Ic
####################################################

Ic=data.obs.loc[:,['IcSm_all']].values.tolist()
ind = data.obs.index.tolist()
Ic = [[ind[i], Ic[i]] for i in range(len(Ic))]
Ic.sort(key=f.Second, reverse=True)

top10=Ic[:int(len(Ic)/5)]
bot10=Ic[-int(len(Ic)/5):]

topPos=[x[0] for x in top10]
botPos=[x[0] for x in bot10]

cellList = data.obs.index.tolist()
L = []
for i in cellList:
    if i in topPos:
        L.append(0)
    elif i in botPos:
        L.append(1)
    else:
        L.append(np.nan)
data.obs['top/bottom_Ic'] = L

####################################################
## pseudobulk using DESeq2 on top 10% vs. bottom 10% based on Ic
####################################################

counts=counts[cellList, data.var.index.tolist()]
counts.obs['top/bottom_Ic'] = data.obs['top/bottom_Ic']
batch=counts.obs['batch'].unique().tolist()

cells={}
for i in [0,1]:
    df = counts.obs.loc[counts.obs['top/bottom_Ic'] == i,:]
    for b in batch:
        c = df.query(f'batch == "{b}"').index.tolist()
        cells[b + '_' + str(i)] = c


samps=[c for c in cells]
pseudoBulkCounts=np.ones(shape=(len(cells),2000))
for c in cells:
    i = samps.index(c)
    pseudoBulkCounts[i,:] = counts[cells[c],:].X.sum(axis=0)
    
df = pd.DataFrame(pseudoBulkCounts, columns = data.var.index.tolist(), index = samps) 
mData = [0 for x in range(23)] + [1 for x in range(23)]
mDf = pd.DataFrame(mData, columns = ['top/bottom'], index = samps) 
pBulk = DeseqDataSet(df, mDf, design_factors = 'top/bottom')
pBulk.deseq2()

stat_res = DeseqStats(pBulk)
stat_res.summary()

results = stat_res.results_df
sigResults = results.query('padj < .05').sort_values(by = 'padj')
sigResults.to_csv('./deseq2_results_ic.csv')


