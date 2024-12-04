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
import json
import kmapper as km
import networkx as nx
import seaborn as sns

data = scp.read_h5ad('./data.h5ad') #use output of compute_diff_map_and_cluster.py
counts = scp.read_h5ad('./counts.h5ad') #use original counts

## NOTE WE INITIALLY TRIED TO USE DESEQ2 FOR THIS
## BUT WE GOT BARELY ANY SIGNIFICANT RESULTS
## SO WE JUST USED THE DEFAULT SCANPY METHOD WITH FILTERING


data.obs['leiden'] = data.obs['leiden'].astype('category')
scp.tl.rank_genes_groups(data, groupby = 'leiden')

geneFrac = []
for i in range(9): 
    cells = data.obs.query(f'leiden == {i}').index.tolist()
    datRed = counts[cells,:].X
    tot = datRed.shape[0]
    nonzero = np.count_nonzero(datRed, axis=0)
    geneFrac.append(nonzero/tot)
    
frac = pd.DataFrame(geneFrac, index = [f'cluster {i}' for i in range(9)], columns=data.var.index).T

results = []
for j in range(9):    
    query = 'pvals_adj < .01 & fraction > .5'
    df = scp.get.rank_genes_groups_df(data, group=str(j))
    df = df.join(frac.loc[:,f'cluster {j}'], on = 'names').query(query).rename(columns = {f'cluster {j}' : 'fraction'})
    results.append(df)


############################################################
## Look for refinement based on differential expression overlap
## Note we exclude the mitochondira since they shows up as up up regulated
## in all clusters except for one in which they are very, very down regulated.  
############################################################

genesTot = [[y for y in x.query('fraction > .5')['names'].tolist() if f.like(y, "MT-.*") == False]  for x in results] 
preDf = {} 
for i in range(9):
    g1 = genesTot[i]
    tot = len(g1)
    preDf[f'cluster{i}'] = []
    for j in range(9):
        g2 = genesTot[j]
        inter = list(set(g1).intersection(g2))
        preDf[f'cluster{i}'].append(len(inter)/tot)

df = pd.DataFrame(preDf, index=[f'cluster{i}' for i in range(9)])
sns.clustermap(df, figsize=(8,6), row_cluster=True, method='average')
plt.savefig('./diffExpGeneOverlap.png')

genesUp=[]
genesDown=[]

genesUpFlat = []
genesDownFlat = []

############################################################
## look at trinarized profiles of diffexp genes   
############################################################

for i in range(9):
    df = results[i]
    up = df.query('fraction > .5 & logfoldchanges>0')['names'].tolist()
    down = df.query('fraction > .5 & logfoldchanges<0')['names'].tolist()
    genesUp.append(up)
    genesDown.append(down)
    genesUpFlat.extend(up)
    genesDownFlat.extend(down)
    
genesUpUnique = list(set(genesUpFlat)) 
counts = []

for g in genesUpUnique:
    c = genesUpFlat.count(g)
    counts.append([g,c])
counts.sort(key=f.Second, reverse = True)
concern = []
for c in counts:
    if c[1]>=4:
        concern.append(c[0])
        con = []
for c in concern:
    lst =  [genesDown.index(d) for d in genesDown if c in d ]
    con.append([c] + lst)
    
genesTot = list(set(genesUpFlat + genesDownFlat))

preDf = {}
for g in genesTot:
    preDf[g] = []
    for i in range(9):
        if g in genesUp[i]:
            preDf[g].append(1)
        elif g in genesDown[i]:
            preDf[g].append(-1)
        else: 
            preDf[g].append(0)
                        
df = pd.DataFrame(preDf, index = [f'cluster{i}' for i in range(9)]).T
sns.clustermap(df, figsize = (8,6), metric = 'jaccard')

############################################################
## we find the same clusters! this motivates the following map  
############################################################
clustMap = {0:2, 1:0, 2:1, 3:3, 4:1, 5:6, 6:4, 7:3, 8:5 }
data.obs['leidenRefined'] = [clustMap[i] for i in data.obs['leiden'].tolist()]

############################################################
## we find the same clusters! this motivates the following map  
############################################################
col = [f'C{i}' for i in data.obs['leidenRefined'].tolist()]
plt.scatter(umap[:,0], umap[:,1], c=col, s=1, alpha =.5)
plt.savfig('./refinedClusterUmap.png')


############################################################
## do it again
############################################################
data.obs['leidenRefined'] = data.obs['leidenRefined'].astype('category')
scp.tl.rank_genes_groups(data, groupby = 'leidenRefined')
geneFrac = []
for i in range(7):
    cells = data.obs.query(f'leidenRefined == {i}').index.tolist()
    datRed = counts[cells,:].X
    tot = datRed.shape[0]
    nonzero = np.count_nonzero(datRed, axis=0)
    geneFrac.append(nonzero/tot)
    
frac = pd.DataFrame(geneFrac, index = [f'cluster {i}' for i in range(7)], columns=data.var.index).T


results = []
for j in range(7):    
    query = 'pvals_adj < .01 & fraction > .5'
    df = scp.get.rank_genes_groups_df(data, group=str(j))
    df = df.join(frac.loc[:,f'cluster {j}'], on = 'names').query(query).rename(columns = {f'cluster {j}' : 'fraction'})
    df.to_csV(f'./refined_clust{i}.csv')
    results.append(df)

genesUp=[]
genesDown=[]

genesUpFlat = []
genesDownFlat = []

for i in range(7):
    df = exp[i]
    up = df.query('fraction > .5 & logfoldchanges>0')['names'].tolist()
    down = df.query('fraction > .5 & logfoldchanges<0')['names'].tolist()
    genesUp.append(up)
    genesDown.append(down)
    genesUpFlat.extend(up)
    genesDownFlat.extend(down)
    
genesUpUnique = list(set(genesUpFlat)) 
counts = []

for g in genesUpUnique:
    c = genesUpFlat.count(g)
    counts.append([g,c])
counts.sort(key=f.Second, reverse = True)

'''
concern = []
for c in counts:
    if c[1]>=4:
        concern.append(c[0])
        con = []
for c in concern:
    lst =  [genesDown.index(d) for d in genesDown if c in d ]
    con.append([c] + lst)
'''   
genesTot = list(set(genesUpFlat + genesDownFlat))

preDf = {}
for g in genesTot:
    preDf[g] = []
    for i in range(7):
        if g in genesUp[i]:
            preDf[g].append(1)
        elif g in genesDown[i]:
            preDf[g].append(-1)
        else: 
            preDf[g].append(0)
            
            
df = pd.DataFrame(preDf, index = [f'cluster{i}' for i in range(7)]).T
df.to_csv('./upDownSummary.csv')
