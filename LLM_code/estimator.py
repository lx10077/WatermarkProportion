import numpy as np
from scipy.optimize import minimize


def inital_estimator(observed_Y, delta=0.1):
    """
    Initial estimator of epsilon (fraction of watermarked data) by assuming 
    null distribution is Uniform(0,1) and ignoring F_P.
    
    Parameters:
        observed_Y (array): Observed (mixed) data matrix.
        delta (float): Threshold for estimation.
    
    Returns:
        float: Estimated epsilon.
    """
    # Estimate: 1 - (empirical CDF at delta) / delta
    return 1 - np.mean(np.mean(observed_Y<=delta, axis=1)/delta)


def refined_estimator_constant_weight(observed_Y, watermarked_Y, delta=0.1):
    """
    Refined estimator of epsilon using an estimate of F_P (watermarked CDF),
    but assuming weight function = 1 (constant).
    
    Parameters:
        observed_Y (array): Observed (mixed) data matrix.
        watermarked_Y (array): Fully watermarked data matrix.
        delta (float): Threshold for estimation.
    
    Returns:
        float: Estimated epsilon.
    """
    F_mix = np.mean(observed_Y <= delta)
    F_p = np.mean(watermarked_Y <= delta)
    return (delta - F_mix) / (delta - F_p)


def histgram_density(data, N=500):
    """
    Estimate the density on [0, 1] using a histogram approach.
    
    Parameters:
        data (array): Input data with support on [0, 1].
        N (int): Number of bins.
    
    Returns:
        tuple: (bin_edges, density estimates)
    """
    data = data.reshape(-1)
    counts, bin_edges = np.histogram(data, bins=N, range=(0, 1), density=True)
    return bin_edges, counts


def evaluate_density(new_data, bin_edges, density):
    """
    Evaluate the histogram-based density for new data points.
    
    Parameters:
        new_data (array): Data points to evaluate.
        bin_edges (array): Bin edges from histogram.
        density (array): Histogram density values.
    
    Returns:
        array: Evaluated density values at new_data points.
    """
    new_data = new_data.reshape(-1)
    bin_indices = np.searchsorted(bin_edges, new_data, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(density) - 1)
    return density[bin_indices]


def refined_estimator_optimal_weight(observed_Y, cdf, use_iterative=False, N0=500):
    """
    Refined estimator of epsilon using optimal weight function.
    
    Parameters:
        observed_Y (array): Observed (mixed) data.
        cdf (array): Fully watermarked sample array.
        use_iterative (bool): Whether to use fixed-point iteration or solver.
        n_sample (int): Number of samples.
        n_legnth (int): Sequence length.
        N0 (int): Number of bins for histogram.
    
    Returns:
        float: Estimated epsilon.
    """
    # Estimate the density of the watermarked distribution
    bin_edges, density = histgram_density(cdf, N=N0)

    # Evaluate densities for uniform baseline, watermark, and mixed
    uniform_data = np.random.uniform(0, 1, 10 ** 6).reshape(-1)
    uniform_kde = evaluate_density(uniform_data, bin_edges, density)
    P_kde = evaluate_density(cdf, bin_edges, density)
    Y_kde = evaluate_density(observed_Y, bin_edges, density)

    # Fixed-point function: maps epsilon to its updated estimate
    def iterative(eps):
        E0 = np.mean((1 - uniform_kde) / ((1 - eps) + eps * uniform_kde))
        Ep = np.mean((1 - P_kde) / ((1 - eps) + eps * P_kde))
        EY = np.mean((1 - Y_kde) / ((1 - eps) + eps * Y_kde))
        next_eps = max(min((E0 - EY) / (E0 - Ep), 1), 1e-3)  # Clamp between [1e-3, 1]
        return next_eps

    if use_iterative:
        # Fixed-point iteration: fast and typically stable
        eps = 0.5
        for _ in range(1000):
            next_eps = iterative(eps)
            eps = next_eps
    else:
        # Use a numerical solver to minimize the residual |eps - iterative(eps)|
        def loss_function(eps):
            eps = eps[0]
            E0 = np.mean((1 - uniform_kde) / ((1 - eps) + eps * uniform_kde))
            Ep = np.mean((1 - P_kde) / ((1 - eps) + eps * P_kde))
            EY = np.mean((1 - Y_kde) / ((1 - eps) + eps * Y_kde))
            target = max(min((E0 - EY) / (E0 - Ep), 1), 1e-3)
            return np.abs(eps - target)
        
        # Use warm-start with pre-run fixed-point iteration
        inital_eps = 0.9
        for _ in range(20):
            inital_eps = iterative(inital_eps)

        result = minimize(loss_function, inital_eps, bounds=[(0, 1)])
        eps = result.x[0]

    return eps
