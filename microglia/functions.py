import numpy as np
import matplotlib
import scipy.stats as stats
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import pynndescent as pynn
import re
import scipy as spy

def First(val):
    return val[0]
def Second(val): 
    return val[1]
def Third(val): 
    return val[2]
def Fourth(val): 
    return val[3]
pd.set_option('display.max_rows', 200)


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


def partition(lst, n):
#    random.shuffle(lst)
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


def like(x, pattern):
    r = re.compile(pattern)
    vlike = np.vectorize(lambda val: bool(r.fullmatch(val)))
    return vlike(x)

def getTransitionMatrix(W):
    #get SDE normalization of kernel
    # t gives number of time steps you wish to investigate
    q = np.asarray(W.sum(axis=0))
    q = np.sqrt(q)
#    q[q==0] = 1
    if not spy.sparse.issparse(W):
        Q = np.diag(1.0 / q)
    else:
        Q = spy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
 #   Q[Q==1] = 0
    K = Q @ W @ Q

    return K


def getDiffMap(K,t=1):
    #get SDE normalization of kernel
    # t gives number of time steps you wish to investigate
    c=K.sum(axis=0)
    C=spy.sparse.spdiags(1.0 / c, 0, K.shape[0], K.shape[0])
    T= K @ C
    eVal, eVec=spy.sparse.linalg.eigs(T, k=200)
    eVal, eVec=np.real(eVal), np.real(eVec)
    E=np.diag(eVal)
    E=np.linalg.matrix_power(E, t)
    
    dMap=eVec @ E
    
    return dMap, eVal


def plot_diffmap(diff_map, dim1 = 1, dim2 = 2, dim3 = 3, c = None, elev = 30, azim = 30, **kwargs):
    fig = plt.figure(figsize=(12,10), dpi=90)
    if c is None:
        col = None
    else:
        col = c
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diff_map[:,dim1], diff_map[:,dim2], diff_map[:,dim3], c = col, s=10, alpha =1, marker = 'o')
    ax.set_xlabel(f'DC{dim1}')
    ax.set_ylabel(f'DC{dim2}')
    ax.set_zlabel(f'DC{dim3}')
    elev = elev
    azim = azim
    ax.view_init(elev, azim)
    plt.tight_layout()
    plt.show()



def test(x):
    v= [i*x for i in range(10)]
    df = pd.DataFrame(v)
    return df




	

