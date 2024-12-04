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
import functions as f
from time import time
from sklearn.cluster import DBSCAN
import json
import kmapper as km
import networkx as nx 
import seaborn as sns
from adjustText import adjust_text
import igraph as ig
import leidenalg as la
import fastcluster
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

###### FUNCTIONS ############

def nn_graph_directed(ind, dist, bandwith = 'max' ):
    # NN graph with Gaussian kernel and bandwidth = kth NN    
    nnGraph = np.zeros(shape=(len(dist), len(dist)))
    if bandwith =='max':
        sig = np.max(dist[:,1:], axis=1)
    elif bandwidth == 'min':
        sig = np.min(dist[:,1:], axis=1)
    elif bandwith == 'mean':
        sig = dist[:,1:].mean(axis=1)
    elif bandwith =='median':
        sig = np.median(dist[:,1:], axis=1)

    for i in range(len(dist)):
        nnGraph[ind[i], i] = np.exp( - (dist[i]**2)/(sig[i]**2))
        nnGraph[i,i] = 0
        
def plot_diffmap(diff_map, dim1 = 1, dim2 = 2, dim3 = 3, c = None, elev = 30, azim = 30,**kwargs):
    fig = plt.figure(figsize=(12,10), dpi=90)
    if c is None:
        col = None
    else:
        col = c
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diff_map[:,dim1], diff_map[:,dim2], diff_map[:,dim3], c = col, **kwargs)
    ax.set_xlabel(f'DC{dim1}')
    ax.set_ylabel(f'DC{dim2}')
    ax.set_zlabel(f'DC{dim3}')
    elev = elev
    azim = azim
    ax.view_init(elev, azim)
    plt.tight_layout()
    plt.show()



########### LOAD DATA ############
data = scp.read_h5ad('../PATH/TO/PROCESSED/DATA') 
count = scp.read_h5ad('../PATH/TO/FILES/') 
genes = data.var.index.tolist()
ikeep = data.obs.index.tolist()
count = count[ikeep,genes]

## project onto PCs. Can vary the number if you want
scp.pp.pca(data, n_comps=30)
v=data.uns['pca']['variance']
x=[x for x in range(30)] ### could potentially drop to 15 or 10 components.
plt.scatter(x,v)
plt.show()
#plt.savefig('./screePlot.png')

#### find nearest neighbors. make sure to use PCs for this
X=data.obsm['X_pca']
I=pynn.NNDescent(X, n_neighbors=30)
ind,dist=I.query(X,k=30)


nn = nn_graph_directed(ind, dist)
W = getTransitionMatrix(nn)
dMap, eVals = getDiffMap(W)

data.obsm['X_diffmap'] = dMap
data.uns['diffmap_evals'] = eVals

###### PLOT EIGENVALUES TO DETERMINE CUTOFF ########
eVals = data.uns['diffmap_evals']
plt.scatter([i for i, e in enumerate(eVals)], eVals, s = 5)
## we choose 10 ##

####### HIERARCHICAL CLUSTERING ON DIFFUSION MAPS #########
dMap = data.obsm['X_diffmap']
X = dMap[:, 1:10] 
dist = pdist(X)
link = fastcluster.linkage(dist, method = 'ward')

X = np.linspace(0,2,1_000)
Y = []
for t in X:
    clust = fcluster(link, t = t, criterion = 'distance')
    Y.append(max(clust))

plt.scatter(X[100:600],Y[100:600])
plt.ylabel('number of clusters', fontsize = 14)
plt.xlabel('max intra-cluster distance', fontsize = 14)
#plt.savefig('/YOUR/PATH/HERE/', dpi = 300)

clusters = fcluster(link, t = .7, criterion = 'distance')
#list(zip(*np.unique(Y, return_counts = True)))
data.obs['clust2_old'] = [x-1 for x in clusters]

####### MAKE SUPPLEMENTARY FIGURE 3b ###########
fig = plt.figure(figsize=(8,6), dpi=90)
ax = plt.axes()
umap = data.obsm['X_umap']
ax.scatter(umap[:,0], umap[:,1], c = clust_col, s = 5, alpha = .5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.spines['bottom'].set_visible(False)

plt.xticks([])
plt.yticks([])
plt.xlabel('umap$_1$', fontsize = 18)
plt.ylabel('umap$_2$', fontsize=18)
#plt.savefig(f'/YOUR/PATH/HERE', dpi = 300)


########## PSEUDOBULK EXPRESSION ########
n_clust = max(clusters)
batch = data.obs['batch'].unique().tolist()
clust = [i for i in range(n_clust)]
countsByBatchClust={}

geneFrac = []
for i in range(n_clust):
    cells = data.obs.query(f'clust2_old == {i}').index.tolist()
    datRed = count[cells,:].X
    tot = datRed.shape[0]
    nonzero = np.count_nonzero(datRed, axis=0)
    geneFrac.append(nonzero/tot)
    
frac = pd.DataFrame(geneFrac, index = [f'cluster_{i}' for i in range(n_clust)], columns=data.var.index).T

for c in clust:
    keep = data.obs.query(f'clust2_old =={c}')
    for b in batch:
        cells = keep.query(f'batch == "{b}"').index.tolist()
        if len(cells) == 0:
            dat = np.zeros(shape=(2000,))
        else:
            dat = count[cells,:].X.sum(axis=0).tolist()
        countsByBatchClust[f'{b}_{c}'] = dat
pBulkDf=pd.DataFrame(countsByBatchClust)    
pBulkDf.index = genes
df = pBulkDf.T

md = np.zeros(shape = (len(batch)*len(clust),n_clust))
for i in range(n_clust):
        md[len(batch)*i:len(batch)*(i+1),i] = np.ones(shape = (len(batch)))
        
mdf=pd.DataFrame(md, columns = [f'cluster_{x}' for x in range(n_clust)], index = [x for x in countsByBatchClust])

s = df.sum(axis=1)
drop = s.where(s == 0).dropna().index.tolist()
keep = [x for x in df.index.tolist() if x not in drop]

df = df.loc[keep,:].astype(int)
mdf = mdf.loc[keep,:]

results_pb = []
for i,c in enumerate(mdf.columns.tolist()):
    pBulk = DeseqDataSet(counts = df, metadata = mdf, design_factors = c)
    pBulk.obs = pBulk.obs.astype(np.float64)
    pBulk.deseq2()

    stat_res = DeseqStats(pBulk)
    stat_res.summary()

    res = stat_res.results_df
    res = res.join(frac.loc[:,f'cluster_{i}'])
    res = res.rename({f'cluster_{i}' : 'fraction'}, axis = 1)
    sigResults = res#.query('padj < .05').sort_values(by = 'padj')
    results_pb.append(sigResults)
    
sig_genes = []
for df in results_pb:
    sig_genes.extend(df.query('(padj < .05)&(fraction > .5)').index)
    
sig_genes = list(set(sig_genes))

###### MAKE SUPPLEMENTARY FIGURE 3C ##########
cols = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'gold', 'light blue', 'black']
preDf = {}
for c in clust:
    cells = data.obs.query(f'clust2_old == {c}').index.tolist()
    preDf[cols[c]] = np.array(data[cells, sig_genes].X.mean(axis=0))[0]

df_sig = pd.DataFrame(preDf, index = sig_genes)
g = sns.clustermap(df_sig,z_score =0, col_cluster = True, row_cluster = True)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, fontsize = 14)
g.ax_row_dendrogram.set_visible(False)
#plt.savefig('/YOUR/PATH/HERE/', dpi = 300)

######### MERGE CLUSTERS BASED ON EXPRESSION SIGNATURE ############
agg_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:3, 6:5, 7:3, 8:6, 9:7, 10:8}
clusters = np.vectorize(agg_map.__getitem__)(clusters) #### the result of running, then merging clusters based on differential expression
data.obs['clust2'] = clusters 
clusters = data.obs['clust2']
n_clust = max(clusters)
batch = data.obs['batch'].unique().tolist()
clust = [i for i in range(n_clust)]
countsByBatchClust={}

geneFrac = []
for i in range(n_clust):
    cells = data.obs.query(f'clust2_old == {i}').index.tolist()
    datRed = count[cells,:].X
    tot = datRed.shape[0]
    nonzero = np.count_nonzero(datRed, axis=0)
    geneFrac.append(nonzero/tot)
    
frac = pd.DataFrame(geneFrac, index = [f'cluster_{i}' for i in range(n_clust)], columns=data.var.index).T


for c in clust:
    keep = data.obs.query(f'clust2_old =={c}')
    for b in batch:
        cells = keep.query(f'batch == "{b}"').index.tolist()
        if len(cells) == 0:
            dat = np.zeros(shape=(2000,))
        else:
            dat = count[cells,:].X.sum(axis=0).tolist()
        countsByBatchClust[f'{b}_{c}'] = dat
pBulkDf=pd.DataFrame(countsByBatchClust)    
pBulkDf.index = genes
df = pBulkDf.T

md = np.zeros(shape = (len(batch)*len(clust),n_clust))
for i in range(n_clust):
        md[len(batch)*i:len(batch)*(i+1),i] = np.ones(shape = (len(batch)))
        
mdf=pd.DataFrame(md, columns = [f'cluster_{x}' for x in range(n_clust)], index = [x for x in countsByBatchClust])

s = df.sum(axis=1)
drop = s.where(s == 0).dropna().index.tolist()
keep = [x for x in df.index.tolist() if x not in drop]

df = df.loc[keep,:].astype(int)
mdf = mdf.loc[keep,:]

results_pb = []
for i,c in enumerate(mdf.columns.tolist()):
    pBulk = DeseqDataSet(counts = df, metadata = mdf, design_factors = c)
    pBulk.obs = pBulk.obs.astype(np.float64)
    pBulk.deseq2()

    stat_res = DeseqStats(pBulk)
    stat_res.summary()

    res = stat_res.results_df
    res = res.join(frac.loc[:,f'cluster_{i}'])
    res = res.rename({f'cluster_{i}' : 'fraction'}, axis = 1)
    sigResults = res#.query('padj < .05').sort_values(by = 'padj')
    results_pb.append(sigResults)

to_concat = []
for i, df in enumerate(results_pb):
    df1 = df.query('(padj < .05)&(fraction > .5)')
    df1.loc[:,['cluster']] = [i for j in range(len(df1))]
    to_concat.append(df1)
    
df_tot = pd.concat(to_concat)
df_tot.to_csv('./new_clusters_pseudobulk.csv')

