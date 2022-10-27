#!/usr/bin/python
import pandas as pd
import numpy as np



def aprox_derv_forward(data,step_size):
    ''' Forward approximation of the derivative a time series.
    The resulting set (derv) will be size n-stepsize.
    Args:
        data -- time series data of which you are approximating the derivative of.
        step_size -- '''
    derv = []
    for i in range(len(data)-step_size):
        val = round(((data[i+step_size]-data[i])/step_size),2)
        derv.append(val)
    return derv

def aprox_derv_backward(data,step_size):
    derv = []
    for i in range(len(data)):
        val = (data[i]-data[i-step_size])/step_size
        derv.append(val)
    return derv

def aprox_derv_central(data,step_size_forward,step_size_backward):
    derv = []
    if step_size_backward<0:
        raise ValueError()
    for i in range(len(data)):
        try:
            if i-step_size_backward<0:
                val = float("NAN")
            else:
                val = (data[i+step_size_forward]-data[i-step_size_backward])/(step_size_forward+step_size_backward)
        except IndexError:
            val = float("NAN")
        derv.append(val)
    return derv

def aprox_derv_forward_v2(data,step_size):
    return np.diff(data,1)/step_size

def savitzky_golay(y_list, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        y = np.asarray(y_list)
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def inflection_point_classifaction(df,sensitivity=0.1):
    classifacation = []
    max_id=df['x'].idxmax()
    for index, row in df.iterrows():
        try:
            no_key_error=bool((index+1)<=max_id)
            index_not_zero=bool(index!=0)
            if no_key_error and index_not_zero:
                sig_max_check=abs(df.loc[int(index+1), 'yx_dx2'])>sensitivity
                sig_min_check=abs(df.loc[int(index+1), 'yx_dx2'])>sensitivity
                if round(row['yx_dx'],2)==0.0 and round(row['yx_dx2'],2)>0.0 and sig_min_check:
                    val='min'
                elif round(row['yx_dx'],2)==0.0 and round(row['yx_dx2'],2)<0.0 and sig_max_check:
                    val='max'
                elif df.loc[int(index+1), 'yx_dx']>0 and df.loc[int(index), 'yx_dx']<0 and sig_min_check:
                    val='min'
                elif df.loc[int(index+1), 'yx_dx']<0 and df.loc[int(index), 'yx_dx']>0 and sig_max_check:
                    val='max'
                else:
                    val=float('NAN')
            else:
                val=float('NAN')
        except KeyError:
            val=float('NAN')
        classifacation.append(val)
    return classifacation

def plotable_points(df):
    dot_class = []
    for index, row in df.iterrows():
        if row['point_classifacation'] in ['min','max']:
            val=row['y']
        else:
            val=float('NAN')
        dot_class.append(val)
    return dot_class

def min_max_flip_by_gen(df):
    min_max = []
    cur_val=None
    for index, row in df.iterrows():
        if row['point_classifacation'] == 'min':
            cur_val=0
        elif row['point_classifacation'] == 'max':
            cur_val=1
        else:
            cur_val=cur_val
        min_max.append(cur_val)
    return min_max

def min_max_filter(df):
    filter_vals = []
    max_id=df['x'].idxmax()
    for index, row in df.iterrows():
        try:
            if df.loc[int(index+1), 'point_classifacation'] in ['min','max']:
                val=1
            else:
                val=0
        except KeyError:
            val=0
        filter_vals.append(val)
    return filter_vals

def time_as_min_max(df):
    time_vals = []
    for index, row in df.iterrows():
        try:
            if df.loc[int(index), 'point_classifacation'] in ['min','max']:
                val=1
            elif index+1 < len(df) and index - 1 >= 0 and time_vals[index - 1]>=0:
                val=time_vals[index - 1]+1
            else:
                val=float('NaN')
        except KeyError:
            val=float('NaN')
        time_vals.append(val)
    return time_vals