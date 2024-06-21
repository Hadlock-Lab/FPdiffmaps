import scanpy as scp
#NOTE: I CHANGED THE DIFFMAP NORMALIZATION TO THE SDE ONE
import anndata as ad 
#import scvelo as scv
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.sparse as sparse
import scipy.cluster as clust
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import re
from sklearn.neighbors import NearestNeighbors as kNN
import pynndescent as pynn
import gget
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import OneHotEncoder
import shap
import functions as f

def First(val):
    return val[0]
def Second(val): 
    return val[1]
def Third(val): 
    return val[2]
def Fourth(val): 
    return val[3]
pd.set_option('display.max_rows', 200)


###################################################
## load things in
###################################################
dfSigGenes=pd.read_csv('./upDownSummary.csv', index_col=0) ## should come out of differential expression script
genes=dfSigGenes.index.to_list()
data=ad.read_h5ad('./data.h5ad') # choose appropriate path
dataUse=data[:,genes]
param = np.load('./param_dict.npy', allow_pickle=True)

###################################################
## prepare the data and run model
###################################################
y=data.obs['leidenRefined'].tolist()
Y=np.zeros(shape=(len(y),1))
Y[:,0]=y   
enc = OneHotEncoder()
enc.fit_transform(Y).todense()

X=np.array(dataUse.X.todense())
X=sts.zscore(X, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=69)

model = xgb.XGBClassifier(param, objective='multi:softprob', eval_metric = 'mlogloss')
model.fit(X_train, Y_train, verbose=1)
Y_pred=model.predict(X_test)
Y_predProba=model.predict_proba(X_test)
cm = confusion_matrix(Y_test, Y_pred, normalize='true')

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True, cmap='jet')
plt.savefig('.confusion_matrix.png')

###################################################
## perform SHAP analysis
###################################################

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names = genes, max_display=30)

for i in range(7):
    plt.figure(figsize = (10,8))
    shap.summary_plot(shap_values[i], X_test, feature_names = genes, max_display=25, plot_size=[10,8], show = False)
    plt.savefig(f'./shapFigs/shapFigs{i}.png')

###################################################
## calculate overlap with differentially expressed genes
###################################################
num = 25
shapGenes = []
for i in range(7):
    maxx = shap_values[i].max(axis=0)
    ind = np.argpartition(maxx, I-num)[-num:]
    ind = ind[np.argsort(maxx[ind])]
    g = [genes[j] for j in ind]
    shapGenes.append(g)

sigGenes = []
for i in range(7):
    gShap = shapGenes[i]
    gDE = dfSigGenes.query(f'cluster{i} != 0').index.tolist()
    
    overlap = list(set(gShap).intersection(gDE))
    sigGenes.append(overlap)

with open("genesFinal", "wb") as fp:   
   pickle.dump(sigGenes, fp)



   
