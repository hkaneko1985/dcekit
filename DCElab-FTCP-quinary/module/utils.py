# Modified from utils.py of PV-Lab/FTCP (https://github.com/PV-Lab/FTCP),
# licensed under the Apache License 2.0. Modified in this work.

import numpy as np

from sklearn.preprocessing import MinMaxScaler


def pad(FTCP, pad_width):
    '''
    This function zero pads (to the end of) the FTCP representation along the second dimension

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation as numpy ndarray.
    pad_width : int
        Number of values padded to the end of the second dimension.

    Returns
    -------
    FTCP : numpy ndarray
        Padded FTCP representation.

    '''
    
    FTCP = np.pad(FTCP, ((0, 0), (0, pad_width), (0, 0)), constant_values=0)
    return FTCP

def minmax(FTCP):
    '''
    This function performs data normalization for FTCP representation along the second dimension

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation as numpy ndarray.

    Returns
    -------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler used for the normalization.

    '''
    
    dim0, dim1, dim2 = FTCP.shape
    scaler = MinMaxScaler()
    FTCP_ = np.transpose(FTCP, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0*dim2)
    FTCP_ = scaler.fit_transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP_normed = np.transpose(FTCP_, (1, 0, 2))
    
    return FTCP_normed, scaler

def minmax_fit(FTCP):
    '''
    Fits a MinMaxScaler on the given FTCP representation WITHOUT transforming it.
    Use this on the training subset only, then reuse the returned scaler with
    minmax_transform() on the training/validation/test subsets, so that no
    information from the validation or test data leaks into the normalization
    parameters (min/max per column).

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation (typically the training subset only).

    Returns
    -------
    scaler : sklearn MinMaxScaler object
        MinMaxScaler fitted on FTCP, not yet applied to any data.
    '''

    dim0, dim1, dim2 = FTCP.shape
    scaler = MinMaxScaler()
    FTCP_ = np.transpose(FTCP, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0 * dim2)
    scaler.fit(FTCP_.T)

    return scaler


def minmax_transform(FTCP, scaler):
    '''
    Applies an already-fitted MinMaxScaler (see minmax_fit) to FTCP data
    without refitting it. Use this for validation/test subsets (and for the
    training subset itself) after fitting the scaler on the training data
    only, to avoid validation/test information leaking into the
    normalization parameters.

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation to normalize.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler previously fitted via minmax_fit (or minmax).

    Returns
    -------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    '''

    dim0, dim1, dim2 = FTCP.shape
    FTCP_ = np.transpose(FTCP, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0 * dim2)
    FTCP_ = scaler.transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP_normed = np.transpose(FTCP_, (1, 0, 2))

    return FTCP_normed


def inv_minmax(FTCP_normed, scaler):
    '''
    This function is the inverse of minmax, 
    which denormalize the FTCP representation along the second dimension

    Parameters
    ----------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler used for the normalization.

    Returns
    -------
    FTCP : numpy ndarray
        Denormalized FTCP representation as numpy ndarray.

    '''
    dim0, dim1, dim2 = FTCP_normed.shape

    FTCP_ = np.transpose(FTCP_normed, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0*dim2)
    FTCP_ = scaler.inverse_transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP = np.transpose(FTCP_, (1, 0, 2))
    
    return FTCP