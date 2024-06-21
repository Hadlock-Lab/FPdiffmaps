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

data = scp.read_h5ad('./data.h5ad') #put in the correct path
bulkData = pd.read_csv('./BulkSortedCNZ-data.tsv', sep = '\t') #availible at synapse #XXXXXXX
bulkMD = pd.read_csv('./BulkSortedCNZ-metadata.tsv', sep = '\t') #availible at synapse #XXXXXXX

data.X = data.X.expm1()
genesUse = data.var.index.tolist()

with open("genesFinal", "rb") as fp:   
   sigGenes = pickle.load(fp)

sigGenesFlat = []
for x in sigGenes:
    sigGenesFlat.extend(x)
unqSig=list(set(sigGenesFlat))

sig = dfSigGenes.loc[unqSig,:]


sigMat = np.zeros(shape=(len(unqSig,7))
for i in range(7):
    cells = data.obs.query(f'leidenRefined == {i}').index.tolist()
    sigMat[:,i] = np.array(data[cells,unqSig].X.mean(axis=0))[0]
                  


sns.clustermap(sigMat, z_score = 0, yticklabels = unqSig) #visualize it
dfSigMatMean = pd.DataFrame(sigMat, columns = [f'cluster{i}' for i in range(7)], index = unqSig)
dfSigMatMean.to_csv('./signatureMatrixMean.csv')

con = bulkMD.query('control == 1').index.tolist()
ad = bulkMD.query('ad == 1').index.tolist()

annot = scp.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", "hgnc_symbol","chromosome_name", "start_position", "end_position"],
    ).set_index("ensembl_gene_id")
                  
dat = bulkData.join(annot).query('hgnc_symbol in @genesUse').set_index('hgnc_symbol')

datCon = dat[con].loc[genesUse,:]
datAd = dat[ad].loc[genesUse,:]

datConNorm =datCon.divide(datCon.sum(axis=0), axis=1)*10000
datAdNorm =datAd.divide(datAd.sum(axis=0), axis=1)*10000

dataConNorm.to_csv('./bulk_control_CPTT.csv')
dataAdNorm.to_csv('./bulk_ad_CPTT.csv')
                   


                  

