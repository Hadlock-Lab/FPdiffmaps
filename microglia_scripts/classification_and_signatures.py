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
from matplotlib.axis import Axis
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
import matplotlib.cm as cmx
from functions import partition, like, scatter3d, getDiffMap

###################### NECESSARY FUNCTIONS #################
def partition(lst, n):
#    random.shuffle(lst)
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


def like(x, pattern):
    r = re.compile(pattern)
    vlike = np.vectorize(lambda val: bool(r.fullmatch(val)))
    return vlike(x)

def scatter3d(x,y,z, cs, colorsMap='jet', size=(12,10), dpi=90):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=2)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

def getDiffMap(W,t=1):
    #get SDE normalization of kernel
    # t gives number of time steps you wish to investigate
    q = np.asarray(W.sum(axis=0))
    q = np.sqrt(q)
    if not spy.sparse.issparse(W):
        Q = np.diag(1.0 / q)
    else:
        Q = spy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
    K = Q @ W @ Q
    
    c=K.sum(axis=0)
    C=spy.sparse.spdiags(1.0 / c, 0, W.shape[0], W.shape[0])
    T= K @ C
    
    eVal, eVec=spy.sparse.linalg.eigs(T, k=100)
    eVal, eVec=np.real(eVal), np.real(eVec)
    E=np.diag(eVal)
    E=np.linalg.matrix_power(E, t)
    
    dMap=eVec @ E
    
    return dMap, eVal

###################### ANALYSIS #################

dfSigGenes=pd.read_csv('YOUR/PATH/TO/SIGNATURE/GENES/HERE/', index_col=0)
genes=list(set(dfSigGenes.query('fraction > .5').index.to_list()))
data=ad.read_h5ad('../data_8_25_23.h5ad')
dataUse=data[:,genes]

##### from hyperparameter tuning #######
param = {'subsample': 0.7,  
 'reg_lambda': 0.001,
 'reg_alpha': 1,
 'n_estimators': 1000,
 'min_child_weight': 3,
 'max_depth': 7,
 'learning_rate': 0.01,
 'gamma': 0.5,
 'colsample_bytree': 0.8} 


######## PREPARE THE DATA ########
y=data.obs['clust2'].tolist()
Y=np.zeros(shape=(len(y),1))
Y[:,0]=y   

X=np.array(dataUse.X.todense())
X=sts.zscore(X, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=55)

####### TRAIN THE MODEL ########
model = xgb.XGBClassifier(param, objective='multi:softprob', eval_metric = 'mlogloss')
model.fit(X_train, Y_train, verbose=1)
Y_pred=model.predict(X_test)
Y_predProba=model.predict_proba(X_test)
cm = confusion_matrix(Y_test, Y_pred, normalize='true')

####### MAKE FIGURE FOR CONFUSION MATRIX #######
clust_names = ['anti\ninflammatory', 'exAM', 'cytokine', 'antigen\npresenting','DAM', 'homeostatic', 'transitional', 'metbolically\nstagnant', 'metabolically\nactive']

plt.figure(figsize=(10,8))

g = sns.heatmap(cm,annot=True, cmap='jet', xticklabels = clust_names, yticklabels = clust_names, cbar = False)
plt.xticks(rotation = 40)
#plt.savefig('/YOUR/PATH/HERE/', dpi = 300)

########## MAKE AUC PLOTS ########

y_pred = Y_pred
y_proba = Y_predProba
n_class = len(y_proba[0])
curves_out = []
scores_out = []
for i in range(n_class):
    y_pred_bin = np.array([1 if c == i else 0 for c in y_pred])
    y_proba_bin = np.array([c[i] for c in y_proba])
    auc = roc_auc_score(y_pred_bin, y_proba_bin)
    roc = roc_curve(y_pred_bin, y_proba_bin)

    curves_out.append(roc)
    scores_out.append(auc)

curves, scores = roc_curve_multiclass(Y_test, Y_predProba)


fig = plt.figure(figsize = (8,6))
ax = plt.axes()
for i, c in enumerate(curves):
    ax.plot(c[0], c[1], c = f'C{i}', label = clust_names[i])
ax.plot(np.linspace(0,1,10), np.linspace(0,1,10), c = 'black', linestyle = '--')
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
#ax.grid(False)
plt.xticks(fontsize = 10)
plt.title('ROC curves for one vs. all classification', fontsize = 14)
plt.xlabel('FPR', fontsize = 12)
plt.ylabel('TPR', fontsize = 12)
plt.legend()
#plt.savefig('/YOUR/PATH/HERE', dpi = 300)

########### CALCULATE SHAP ############
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

balanced_acc = cm.trace()/cm.shape[0]

######## MAKE FIGURE 4C
for i in range(9):
    if i in [1,2,4]: ### for specific formatting 
        p = shap.summary_plot(shap_values[i], X_test, feature_names = genes, max_display=35, plot_size=[10,8], show = False)
        plt.title(clust_names[i], fontsize = 20)
        plt.tight_layout()
#        plt.savefig(f'/YOUR/PATH/HERE')
        plt.show()
    else:
        p = shap.summary_plot(shap_values[i], X_test, feature_names = genes, color_bar = False, max_display=35, plot_size=[10,8], show = False)
        plt.title(clust_names[i], fontsize = 20)
        plt.tight_layout()
#        plt.savefig('/YOUR/PATH/HERE')
        plt.show()

######### IDENTIFY SHAP GENES TO USE ########
num = 35 ## 
shapGenes = []
for i in range(9):
    maxx = shap_values[i].var(axis=0)
    ind = np.argpartition(maxx, -num)[-num:]
    ind = ind[np.argsort(maxx[ind])]
    g = [genes[j] for j in ind]
    shapGenes.append(g)
    
sigGenes = []
for i in range(9):
    gShap = shapGenes[i]
    gDE = dfSigGenes.query(f'cluster == {i}').index.tolist()
    
    overlap = list(set(gShap).intersection(gDE))
    sigGenes.append(overlap)
    
sigGenesFlat = []
for x in sigGenes:
    sigGenesFlat.extend(x)
unqSig=list(set(sigGenesFlat))
pd.DataFrame(unqSig, columns = ['signatureGenes']).to_csv('/YOUR/HERE/uniqueSignatureGenes.csv')

############### USE THESE TO BUILT CELL TYPE SIGNATURE MATRIX #################
sigMat = np.zeros(shape=(len(unqSig),9))
for i in range(9):
    cells = data.obs.query(f'clust2 == {i}').index.tolist() #change to whatever you cluster column is
    sigMat[:,i] = np.array(data[cells,unqSig].X.mean(axis=0))[0]
    
dfSigMatMean = pd.DataFrame(sigMat, columns = [f'cluster{i}' for i in range(9)], index = unqSig)
dfSigMatMean.to_csv('./YOUR/PATH/HERE/signatureMatrixMean.csv')

####### MAKE FIRUE 1D ##########
#sns.set(font_scale=.7)
g = sns.clustermap(sigMat, z_score = 0, yticklabels = unqSig, xticklabels = clust_names, row_cluster = True, cbar = False)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, fontsize = 14)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize = 5)
g.ax_row_dendrogram.set_visible(False)
#plt.savefig('/Users/andrew/Desktop/microgliaFigs/signatureHeatmap.png', dpi = 300)
plt.show()

