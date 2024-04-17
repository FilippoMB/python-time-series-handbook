import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm.notebook import tqdm

def _runge_kutta_4th_order(func, initial_value, start_time, end_time, params, stimulus=None):
    """
    Compute the integral of an ODE using the fourth-order, 4-step Runge-Kutta method.

    Parameters:
    ----------
    func : function
        ODE function. Must take arguments like func(t, x, p) where x and t are
        the state and time *now*, and p is a tuple of parameters. If there are
        no model parameters, p should be set to the empty tuple.
    initial_value : array_like
        Initial value for calculation.
    start_time : float
        Initial time for calculation.
    end_time : float
        Final time for calculation.
    params : tuple, optional
        Tuple of model parameters for func.
    stimulus : function or array_like, optional
        Stimulus to be applied to the system. If stimulus is a function, it will be evaluated at
        start_time, (start_time+end_time)/2, and end_time. If stimulus is an array, it should contain the
        stimulus values at start_time, (start_time+end_time)/2, and end_time.

    Returns:
    -------
    step : array_like
        Approximation to the integral.

    Notes:
    ------
    This function uses the fourth-order, 4-step Runge-Kutta method to numerically compute the integral of an ODE.

    Example usage:
    --------------
    >>> def f(t, x, p):
    ...     return x * p[0] + p[1]
    >>> initial_value = 1.0
    >>> start_time = 0.0
    >>> end_time = 1.0
    >>> params = (2.0, 1.0)
    >>> step = RK4(f, initial_value, start_time, end_time, params)
    """
    midpoint_time = (start_time + end_time) / 2.0
    time_interval = end_time - start_time

    if stimulus is None:
        params_at_start = params_at_mid = params_at_end = params
    else:
        try:
            # test if stimulus is a function
            stimulus_at_start = stimulus(start_time)
            stimulus_at_start, stimulus_at_mid, stimulus_at_end = (stimulus, stimulus, stimulus)
        except TypeError:
            #  otherwise assume stimulus is an array
            stimulus_at_start, stimulus_at_mid, stimulus_at_end = stimulus
        params_at_start = (params, stimulus_at_start)
        params_at_mid = (params, stimulus_at_mid)
        params_at_end = (params, stimulus_at_end)

    K1 = func(start_time, initial_value, params_at_start)
    K2 = func(midpoint_time, initial_value + time_interval * K1 / 2.0, params_at_mid)
    K3 = func(midpoint_time, initial_value + time_interval * K2 / 2.0, params_at_mid)
    K4 = func(end_time, initial_value + time_interval * K3, params_at_end)

    step = time_interval * (K1 / 2.0 + K2 + K3 + K4 / 2.0) / 3.0

    return step


def _variational_equation(t, Phi, x, func_jac, p=()):
    """ 
    Compute the time derivative of the variational matrix for a set of differential equations.
    
    Parameters:
    ----------
    t : array_like
        Array of times at which to evaluate the variational equation.
    Phi : array_like
        Array representing the variational matrix.
    x : array_like
        Array representing the system state.
    func_jac : function
        Jacobian of the ODE function.
    p : tuple, optional
        Tuple of model parameters for the ODE function.
        
    Returns:
    -------
    dPhi_dt_flat : array_like
        Array representing the time derivative of the variational matrix.
        
    Notes:
    ------
    The variational equation calculates the time derivative of the variational matrix using the Jacobian of the ODE function.
    
    The variational matrix represents the sensitivity of the system state to initial conditions.
    
    The output is a flattened array representing the time derivative of the variational matrix, which can be used for numerical integration.
    
    Example usage:
    --------------
    >>> t = np.linspace(0, 10, 100)
    >>> Phi = np.eye(num_dimensions, dtype=np.float64).flatten()
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> dPhi_dt = _variational_equation(t, Phi, x, fjac, p)
    """
    num_dimensions = len(x)
    Phi_matrix = np.reshape(Phi, (num_dimensions, num_dimensions))
    dPhi_dt = np.dot(func_jac(t, x, p), Phi_matrix)
    dPhi_dt_flat = dPhi_dt.flatten()
    return dPhi_dt_flat


def _combined_state_equations(t, S, num_dimensions, func, func_jac, p=()):
    """
    Propagates the combined state and variational matrix for a set of differential equations.

    Parameters:
    ----------
    t : array_like
        Array of times over which to propagate the combined state.
    S : array_like
        Array representing the combined state, consisting of the system state and the variational matrix.
    num_dimensions : int
        Number of dimensions of the system state.
    func : function
        ODE function. Must take arguments like f(t, x, p) where x and t are the state and time *now*, and p is a tuple of parameters.
    func_jac : function
        Jacobian of f.
    p : tuple, optional
        Tuple of model parameters for f.

    Returns:
    -------
    S_dot : array_like
        Array representing the time derivative of the combined state.

    Notes:
    ------
    The combined state is represented as a flattened array, where the first num_dimensions elements represent the system state, and the remaining elements represent the variational matrix.

    The variational equation is used to calculate the time derivative of the variational matrix.

    The combined state and variational matrix are propagated using the provided ODE function and Jacobian.

    The output is the time derivative of the combined state, which can be used for numerical integration.

    Example usage:
    --------------
    
    >>> t = np.linspace(0, 10, 100)
    >>> S = np.zeros(num_dimensions + num_dimensions**2)
    >>> S_dot = _combined_state_equations(t, S, num_dimensions, f, fjac, p)
    """
    x = S[:num_dimensions]
    Phi = S[num_dimensions:]
    S_dot = np.append(func(t, x, p), _variational_equation(t, Phi, x, func_jac, p))
    return S_dot


def computeLE(func, func_jac, x0, t, p=(), ttrans=None):
    """
    Computes the global Lyapunov exponents for a set of ODEs using the method described 
    in Sandri (1996), through the use of the variational matrix.
    
    Parameters:
    ----------
    func : function
        ODE function. Must take arguments like func(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model parameters, p should be set to the empty tuple.
    func_jac : function
        Jacobian of func.
    x0 : array_like
        Initial position for calculation. Integration of transients will begin 
        from this point.
    t : array_like
        Array of times over which to calculate LE.
    p : tuple, optional
        Tuple of model parameters for f.
    ttrans : array_like, optional
        Times over which to integrate transient behavior.
        If not specified, assumes trajectory is on the attractor.
    
    Returns:
    -------
    LEs : array_like
        Array of global Lyapunov exponents.
    LE_history : array_like
        Array of Lyapunov exponents over time.
    """

    # Change the function signature to match the required format
    func_ = lambda t, x, p: func(t, x, *p)
    func_jac_ = lambda t, x, p: func_jac(t, x, *p)

    # Initialize variables
    num_dimensions = len(x0)
    num_time_steps = len(t)
    if ttrans is not None:
        num_time_steps += len(ttrans) - 1

    # integrate transient behavior
    Phi0 = np.eye(num_dimensions, dtype=np.float64).flatten()
    if ttrans is not None:
        xi = x0
        for i, (t1, t2) in enumerate(zip(ttrans[:-1], ttrans[1:])):
            xip1 = xi + _runge_kutta_4th_order(func_, xi, t1, t2, p)
            xi = xip1
        x0 = xi

    # start LE calculation
    LE = np.zeros((num_time_steps - 1, num_dimensions), dtype=np.float64)
    combined_state_solution = np.zeros((num_time_steps, num_dimensions*(num_dimensions+1)), dtype=np.float64)
    combined_state_solution[0] = np.append(x0, Phi0)

    for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
        combined_state_temp = combined_state_solution[i] + _runge_kutta_4th_order(
            lambda t, S, p: _combined_state_equations(t, S, num_dimensions, func_, func_jac_, p),
            combined_state_solution[i], t1, t2, p)
        # perform QR decomposition on Phi
        Phi_matrix = np.reshape(combined_state_temp[num_dimensions:], (num_dimensions, num_dimensions))
        Q, R = np.linalg.qr(Phi_matrix)
        combined_state_solution[i+1] = np.append(combined_state_temp[:num_dimensions], Q.flatten())
        LE[i] = np.abs(np.diag(R))

    # compute LEs
    LE_history = np.cumsum(np.log(LE + 1e-10), axis=0) / np.tile(t[1:], (num_dimensions, 1)).T

    LEs = LE_history[-1, :]
    return LEs, LE_history


def plot_bifurcation_diagram(func, func_jac, x0, time_vector, parameters, p_idx, max_time=None):
    """
    Computes and plots the bifurcation diagram for a set of ordinary differential equations (ODEs).

    Parameters:
    ----------
    func : function
        ODE function. Must take arguments like func(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model parameters, p should be set to the empty tuple.
    func_jac : function
        Jacobian of func.
    x0 : array_like
        Initial conditions.
    time_vector : array_like
        Time vector for the integration.
    parameters : array_like
        Range of parameter values to explore.
    p_idx : int
        Index of the parameter to vary in the bifurcation diagram.

    Notes:
    ------
    The ODE function should be defined as a callable that takes arguments for the current state and time, and returns the derivative of the state with respect to time.

    The Jacobian function should be defined as a callable that takes arguments for the current state and time, and returns the Jacobian matrix of the ODE function.

    The initial conditions should be specified as an array-like object.

    The time vector should be an array-like object representing the time points at which to evaluate the ODEs.

    The range of parameter values should be specified as an array-like object.

    The index of the parameter to vary in the bifurcation diagram should be an integer.

    The function will compute the solution of the ODEs for each parameter value in the range, and plot the bifurcation diagram showing the local maxima and minima of the state variables, as well as the maximum Lyapunov exponents as a function of the parameter value.
    """
    maxima_x = []
    minima_x = []
    px_max = []
    px_min = []
    maxima_y = []
    minima_y = []
    py_max = []
    py_min = []
    maxima_z = []
    minima_z = []
    pz_max = []
    pz_min = []
    le_list = []

    for _p in tqdm(parameters):

        solution = solve_ivp(func, [time_vector[0], time_vector[-1]], x0, args=_p, t_eval=time_vector)

        local_max_x, _ = find_peaks(solution.y[0])
        local_min_x, _ = find_peaks(-1*solution.y[0])
        local_max_y, _ = find_peaks(solution.y[1])
        local_min_y, _ = find_peaks(-1*solution.y[1])
        local_max_z, _ = find_peaks(solution.y[2])
        local_min_z, _ = find_peaks(-1*solution.y[2])

        maxima_x.extend(solution.y[0, local_max_x])
        minima_x.extend(solution.y[0, local_min_x])
        px_max.extend([_p[p_idx]] * len(local_max_x))
        px_min.extend([_p[p_idx]] * len(local_min_x))
        maxima_y.extend(solution.y[1, local_max_y])
        minima_y.extend(solution.y[1, local_min_y])
        py_max.extend([_p[p_idx]] * len(local_max_y))
        py_min.extend([_p[p_idx]] * len(local_min_y))
        maxima_z.extend(solution.y[2, local_max_z])
        minima_z.extend(solution.y[2, local_min_z])
        pz_max.extend([_p[p_idx]] * len(local_max_z))
        pz_min.extend([_p[p_idx]] * len(local_min_z))

        LE_time = time_vector if max_time is None else time_vector[:max_time]
        LEs, _ = computeLE(func, func_jac, x0, LE_time, p=_p)
        le_list.append(LEs.max())

        x0 = solution.y[:,-1]

    mle = np.array(le_list)
    pos_idx = np.where(mle > 0)[0]
    neg_idx = np.where(mle < 0)[0]
    _, axes = plt.subplots(4, 1, figsize=(15, 15))
    axes[0].plot(px_max, maxima_x, 'ko', markersize=0.2, alpha=0.3, label="Local maxima")
    axes[0].plot(px_min, minima_x, 'o', color='tab:blue', markersize=0.2, alpha=0.3, label="Local minima")
    axes[0].legend(loc='upper left', markerscale=15)
    axes[0].set_ylabel("x-values")
    axes[1].plot(py_max, maxima_y, 'ko', markersize=0.2, alpha=0.3, label="Local maxima")
    axes[1].plot(py_min, minima_y, 'o', color='tab:blue', markersize=0.2, alpha=0.3, label="Local minima")
    axes[1].legend(loc='upper left', markerscale=15)
    axes[1].set_ylabel("y-values")
    axes[2].plot(pz_max, maxima_z, 'ko', markersize=0.2, alpha=0.3, label="Local maxima")
    axes[2].plot(pz_min, minima_z, 'o', color='tab:blue', markersize=0.2, alpha=0.3, label="Local minima")
    axes[2].legend(loc='upper left', markerscale=15)
    axes[2].set_ylabel("z-values")
    axes[3].plot(parameters[:,p_idx][pos_idx], mle[pos_idx], 'o', color='tab:red', markersize=2.5, alpha=0.5)
    axes[3].plot(parameters[:,p_idx][neg_idx], mle[neg_idx], 'ko', markersize=2.5, alpha=0.5)
    axes[3].set_ylabel("Maximum Lyapunov Exponent")
    axes[3].set_xlabel("Parameter Value")
    axes[3].axhline(0, color='k', lw=.5, alpha=.5)