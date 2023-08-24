from sympy import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def hermite_func(n):
    """
    Calculate the nth Hermite function.
    
    Parameters:
    -----------
    n : int
        The order of the Hermite function.
        
    Returns:
    --------
    hermite : sympy expression
        The nth Hermite function.
    """
    x = symbols('x', real=True)
    c = (2*pi)**n *2 **(n-S(1)/2) * factorial(n)
    hermite = c**(-S(1)/2) * exp(pi*x**2) * Derivative(exp(-2*pi*x**2), (x, n)).doit()
    return hermite

def plot_hermite(n_range, x_range):
    """
    Plot a range of Hermite functions.
    
    Parameters:
    -----------
    n_range : list of int
        List of Hermite function orders to plot.
    x_range : tuple of float
        Range of x values for plotting.
        
    Returns:
    --------
    p1 : matplotlib figure
        The plotted figure containing the Hermite functions.
    """
    x = symbols('x', real=True)
    p1 = plot(show=False, legend=True, xlabel='x', ylabel='')
    for n in n_range:
        p1.extend(plot(hermite_func(n), (x, x_range[0], x_range[1]), show=False, label=f'$h_{{{n}}}$'))
    p1.show()
    return p1  

def hermite_num(n):
    """
    Generate a numerical function for the nth Hermite function.
    
    Parameters:
    -----------
    n : int
        The order of the Hermite function.
        
    Returns:
    --------
    function
        A numerical function representing the nth Hermite function.
    """
    x = symbols('x', real=True)
    return lambdify(x, hermite_func(n))

def zak(f, lamb, xx, gam, kmax, kmin=None):
    """
    Compute the (symbolic) Zak transform of the given function f.
    
    Parameters:
    -----------
    f : SymPy expression
        The function to compute the Zak transform for.
    lamb : SymPy symbol
        Scaling/modular parameter (positive real number).
    xx : float
        Time evaluation point.
    gam : float
        Frequency evaluation point.
    kmax : int
        Maximum value of the summation index.
    kmin : int, optional
        Minimum value of the summation index. Default is -kmax.
        
    Returns:
    --------
    complex
        The computed Zak transform value.
    """
    k = symbols('k', integer=True, negative=False)
    if kmin is None:
        kmin = -kmax
    summation = Sum(f.subs(list(f.free_symbols)[0], lamb*(xx+k)) * exp(-2*pi*I*k*gam), (k, kmin, kmax))
    return sqrt(lamb) * summation.doit()

def num_zak_modular(f_num, lamb, xx, gam, kmax, kmin=None):
    """
    Calculate the (numerical) Zak transform of a given numerical function using modular parameters.
    
    Parameters:
    -----------
    f_num : function
        Numerical function to compute the Zak transform for.
    lamb : numpy array
        Array of scaling/modular parameters (positive real numbers).
    xx : float
        Time evaluation point.
    gam : float
        Frequency evaluation point.
    kmax : int
        Maximum value of the summation index.
    kmin : int, optional
        Minimum value of the summation index. Default is -kmax.
        
    Returns:
    --------
    numpy array
        Array containing the calculated Zak transform values.
    """
    if kmin is None:
        kmin = -kmax
    k = np.linspace(kmin, kmax, kmax - kmin + 1)
    lamb = np.reshape(lamb, (len(lamb), 1))
    zak = np.sqrt(lamb.T) * np.sum(f_num(lamb*(xx+k)) * np.exp(-2.0*np.pi*1.0j*k*gam), axis=1).real
    return zak[0]

def f_zeros(f_num, var_int, delt, *args, **kwargs):
    """
    Find zeros of a numerical function within a given interval.
    
    Parameters:
    -----------
    f_num : function
        Numerical function for which zeros are to be found.
    var_int : tuple of float
        Interval within which zeros are to be located.
    delt : float
        Spacing for the x values in the interval.
    *args, **kwargs:
        Additional arguments to be passed to the numerical function.
        
    Returns:
    --------
    numpy array
        Array containing the calculated zero values.
    """
    x = np.linspace(var_int[0], var_int[1], int(np.ceil((var_int[1] - var_int[0]) / delt)))
    f_eval = f_num(x, *args, **kwargs)
    idx_change = np.where(f_eval[1:] * f_eval[:-1] < -1e-08)[0]
    zero = np.zeros(len(idx_change))
    for i in range(len(idx_change)):
        zero[i] = fsolve(lambda x: f_num(x, *args, **kwargs), x[idx_change[i]])
    return zero    

def zak_modular_zeros(f_num, xx, gam, kmax, delt, lamb_max):
    """
    Calculate the zeros of the Zak transform of a numerical function using modular parameters.
    
    Parameters:
    -----------
    f_num : function
        Numerical function to compute the Zak transform for.
    xx : float
        Time evaluation point.
    gam : float
        Frequency evaluation point.
    kmax : int
        Maximum value of the summation index.
    delt : float
        Spacing for the lamb values.
    lamb_max : float
        Maximum value of the scaling/modular parameter.
        
    Returns:
    --------
    numpy array
        Array containing the calculated zero values.
    """
    lamb = np.linspace(0, lamb_max, int(np.ceil(lamb_max / delt)))
    zak = num_zak_modular(f_num, lamb, xx, gam, kmax)
    idx_change = np.where(zak[0, :-1] * zak[0, 1:] < 0)[0]
    zeros = np.zeros(len(idx_change))
    for i in range(len(idx_change)):
        zeros[i] = fsolve(lambda l: num_zak_modular(f_num, l, xx, gam, kmax)[0], lamb[idx_change[i]])
    return zeros

def plot_zak_modular_hermite(lamb_func, lamb_range, xx, gam, kmax, n_range, symbol):
    """
    Plot the Zak transform of multiple Hermite functions using modular parameters.
    
    Parameters:
    -----------
    lamb_func : function
        Function to compute the scaling/modular parameter lambda.
    lamb_range : tuple of float
        Range of lambda values for plotting.
    xx : float
        Time evaluation point.
    gam : float
        Frequency evaluation point.
    kmax : int
        Maximum value of the summation index.
    n_range : list of int
        List of Hermite function orders to plot.
    symbol : str
        Symbol for the x-axis label.
        
    Returns:
    --------
    p1 : matplotlib figure
        The plotted figure containing the Zak transform of the Hermite functions.
    """
    l = symbols('l', real=True)
    p1 = plot(show=False, legend=True, xlabel=symbol, ylabel='')
    for n in n_range:
        p1.extend(plot(zak(hermite_func(n), lamb_func(l), xx, gam, kmax), (l, lamb_range[0], lamb_range[1]),
                       show=False, label='$Z_{' + latex(lamb_func(symbol)) + '}'+ f'h_{ {n} }$'))
    p1.show()
    return p1

def move_sympyplot_to_axes(p, ax):
    """
    Move a sympy plot to a specified matplotlib axes.
    
    Parameters:
    -----------
    p : sympy.plotting.plot.Plot
        The sympy plot to be moved.
    ax : matplotlib.axes._axes.Axes
        The matplotlib axes to which the sympy plot will be moved.
        
    Author: @ImportanceOfBeingErnest, https://stackoverflow.com/a/46813804/11021886
    """
    backend = p.backend(p)
    backend.ax = ax
    # Fix for > sympy v1.5
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)
