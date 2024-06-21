import numpy as np
import matplotlib
import scipy.stats as stats
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import pynndescent as pynn
import re
import scipy as spy

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import expm, issymmetric

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
    K = Q @ W

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
        
    nn = spy.sparse.csr_matrix(nnGraph)
    return nn




@dataclass
class OUParams:
    alpha: np.ndarray  # n x n matrix of drift coefficients
    gamma: np.ndarray  # n-dim vector of asymptitc means
    beta: np.ndarray  # n-dim vector of diffusion coeff00
        
def OU_process_mv(
    T: Union[int, np.ndarray],
    OU_params : OUParams,
    single_source: bool, default = True,
    X0: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    dW_cov : Optional[np.ndarray] = None,
    dW_mean : Optional[np.ndarray] = None,
) -> np.ndarray:
        """
    Code for generating a multivariate Ornstein-Uhlenbeck process
    with additive noise from a single noise source.
    T: int or array,
        Either an integer specifiying the number of time steps 
        or an array specifying specific time points
        If array: we reccomend rescaling times to be in [0,1]
    OU_params: OUParams, 
        Parameter class including a dxd matrix of "spring constants", 
        a d-dim vector of asymptotic means and a d-dim vector of diffusion coeffs.
    X0: d-dim vector, optional, default = None , 
        A d-dim vector of initial starting points. Returns asymptotic mean if None
    random_state: int, optional, default = None,
        Seed for picking a random state for the calcultion of the nouse
    Returns a dxT array for the process
    """
        a = OU_params.alpha
        b = OU_params.beta
        g = OU_params.gamma
        if (single_source == False)&(dW_cov is not None):
            if (issymmetric(dW_cov) == False):
                raise ValueError('Covariance matrix is not symmetric')
        if type(T) == int:
            t = np.arange(T, dtype=np.float128)
        else:
            t = T
            T = T.shape[0]
        if X0 is None:
            X0 = OU_params.gamma
        if single_source == True:
            dW = get_dW(T, random_state)
        else:
            if dW_mean is None:
                dW_mean = np.zeros(b.shape[0])
            if dW_cov is None:
                dW_cov = np.eye(b.shape[0])
            dW = np.random.multivariate_normal(dW_mean, dW_cov, size = T).T
        if len(b.shape) == 2:
            multi_source_corr = True
        else: 
            multi_source_corr = False
        integral_W = _get_integral_W(t,dW,OU_params, multi_source_corr)
        initial_term = np.zeros(shape = (a.shape[0], T))
        integral_term = np.zeros(shape = (a.shape[0], T))
        for i,tt in enumerate(t):
            exp = expm(-a*tt)
            integral_term[:,i] = exp @ integral_W[:,i]
            initial_term[:,i] = exp @ (X0-g) 
        return initial_term + integral_term

def _get_integral_W(
    t: np.ndarray, dW: np.ndarray, OU_params: OUParams, multi_source_corr: bool
) -> np.ndarray:
    """Integral with respect to Brownian Motion (W), ∫e^(-alpha*s)*beta dW.
    where a is a matrix of drift coeffs and beta is a vector of diffusion coeff
    the exponential is a matrix exponential and it is multiplies beta (as a matrix)
    this allows for arbitrary times, not just integer steps"""
    a = OU_params.alpha
    b = OU_params.beta
    exp_alpha_s_beta= np.zeros(shape = (a.shape[0], t.shape[0]))
    if multi_source_corr == True:
        for i,tt in enumerate(t):
            exp_alpha_s_beta[:,i] = (expm(a*tt) @ b) @ dW[:,i]  
        integral_W = np.cumsum(exp_alpha_s_beta , axis=1)
    else:
        for i,tt in enumerate(t):
            exp_alpha_s_beta[:,i] = (expm(a*tt) @ b)  
        if len(dW.shape) !=2:
            B=exp_alpha_s_beta*dW[None,:]
        else:
            B=exp_alpha_s_beta*dW
        integral_W = np.cumsum(B , axis=1)
    return np.insert(integral_W, 0, np.zeros(shape = b.shape), axis=1)[:,:-1]

def get_corr_dW_matrix(
    T: int,
    n_procs: int,
    rho: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    2D array of n_procs discrete Brownian Motion increments dW.
    Each column of the array is one process.
    So that the resulting shape of the array is (T, n_procs).
        - T is the number of samples of each process.
        - The correlation constant rho is used to generate a new process,
            which has rho correlation to a random process already generated,
            hence rho is only an approximation to the pairwise correlation.
        - Optional random_state to reproduce results.
    """
    rng = np.random.default_rng(random_state)
    dWs: list[np.ndarray] = []
    for i in range(n_procs):
        random_state_i = _get_random_state_i(random_state, i)
        if i == 0 or rho is None:
            dW_i = get_dW(T, random_state=random_state_i)
        else:
            dW_corr_ref = _get_corr_ref_dW(dWs, i, rng)
            dW_i = _get_correlated_dW(dW_corr_ref, rho, random_state_i)
        dWs.append(dW_i)
    return np.asarray(dWs).T


def get_dW(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample T times from a normal distribution,
    to simulate discrete increments (dW) of a Brownian Motion.
    Optional random_state to reproduce results.
    """
    np.random.seed(random_state)
    return np.random.normal(0.0, 1.0, T)


def get_W(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Simulate a Brownian motion discretely samplet at unit time increments.
    Returns the cumulative sum
    """
    dW = get_dW(T, random_state)
    # cumulative sum and then make the first index 0.
    dW_cs = dW.cumsum()
    return np.insert(dW_cs, 0, 0)[:-1]


def _get_correlated_dW(
    dW: np.ndarray, rho: float, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Sample correlated discrete Brownian increments to given increments dW.
    """
    dW2 = get_dW(
        len(dW), random_state=random_state
    )  # generate Brownian icrements.
    if np.array_equal(dW2, dW):
        # dW cannot be equal to dW2.
        raise ValueError(
            "Brownian Increment error, try choosing different random state."
        )
    return rho * dW + np.sqrt(1 - rho ** 2) * dW2


def _get_random_state_i(random_state: Optional[int], i: int) -> Optional[int]:
    """Add i to random_state is is int, else return None."""
    return random_state if random_state is None else random_state + i


def _get_corr_ref_dW(
    dWs: list[np.ndarray], i: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Choose randomly a process (dW) the from the
    already generated processes (dWs).
    """
    random_proc_idx = rng.choice(i)
    return dWs[random_proc_idx]








	

