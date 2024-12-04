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
### attempt to perform this on differentially expressed genes only.
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
ind,dist=I.query(X,k=30) ## find neighbors


IcFin=[]
print('calculating')
for i in range(X.shape[0]): 
    x=X[ind[i],:] ## i labels the cells. 
    Ic=getIc(x)
    IcFin.append([i,Ic])
print('saving')
finDat=pd.DataFrame(data=IcFin, columns=['index', 'Ic'])
finDat.to_csv('./patelOlahIcValsPCA_all.csv')
print('done')


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed
