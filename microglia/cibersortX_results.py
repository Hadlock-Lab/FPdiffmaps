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

deconvAd=pd.read_csv('.CIBERSORTx_ad_cptt.csv') #output from CIBERSORTx
deconvCon=pd.read_csv('./CIBERSORTx_con_cptt.csv') #output from CIBERSORTx

res = []
for i in range(7):
    adVals = deconvAd[f'cluster{i}'].values
    conVals = deconvCon[f'cluster{i}'].values
    k, p = stats.ks_2samp(adVals, conVals)
    res.append([k,p])

for i in range(7):    
    plt.hist(deconvAd[f'cluster{i}'], bins=10, alpha = .5, density = True)
    plt.hist(deconvCon[f'cluster{i}'], bins=10, alpha = .5, density = True)
    plt.title(f'cluster{i}')
    plt.show()

