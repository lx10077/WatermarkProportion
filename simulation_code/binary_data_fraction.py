import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Set global plot settings
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

np.random.seed(121)  # Set seed for reproducibility

# ----------------------------- Data Generation ----------------------------- #
def generate_mixture_data(n, epsilon, mu, gamma):
    """
    Generate data from a mixture of two Bernoulli distributions:
    - null: Bernoulli(gamma)
    - alternative: Bernoulli(mu)

    Parameters:
        n (int): Number of samples.
        epsilon (float): Fraction of samples from alternative.
        mu (float): Mean of alternative Bernoulli.
        gamma (float): Mean of null Bernoulli.

    Returns:
        np.ndarray: Mixture sample.
    """
    null_data = np.random.binomial(1, gamma, size=n)
    alt_data = np.random.binomial(1, mu, size=n)

    # Determine which samples are from the alternative component
    is_from_gamma = np.zeros(n, dtype=int)
    k = max(int(round(epsilon * n)), 1)
    indices = np.random.choice(n, size=k, replace=False)
    is_from_gamma[indices] = 1

    # Select values from appropriate component
    data = np.where(is_from_gamma == 1, alt_data, null_data)
    return data

# ------------------------------ Estimators ------------------------------- #
def empirical_mean(data):
    return np.mean(data)

def empirical_variance(data):
    return np.var(data)

def estimate_eps_mu(data, null_gamma=0.5, lmbd=1e-2):
    """
    Estimate (epsilon, mu) via maximum likelihood, assuming null mean is known.

    Parameters:
        data (np.ndarray): Observed mixture data.
        null_gamma (float): Known null Bernoulli mean.
        lmbd (float): Regularization weight.

    Returns:
        tuple: Estimated (epsilon, mu)
    """
    empirical_mean_value = empirical_mean(data)

    def objective(x):
        epsilon_est, mu_est = x
        theoretical_mean = (1 - epsilon_est) * null_gamma + epsilon_est * mu_est
        # Log-likelihood of empirical mean under Bernoulli
        loglik = empirical_mean_value * np.log(theoretical_mean + 1e-8) + \
                 (1 - empirical_mean_value) * np.log(1 - theoretical_mean + 1e-8)
        # Add small regularization to avoid degenerate solutions
        return -loglik + lmbd * (epsilon_est**2 + mu_est**2)

    result = minimize(objective, (0.1, 0.7), bounds=[(1e-3, 1 - 1e-3), (1e-3, 1 - 1e-3)], tol=1e-5)
    return result.x  # estimated (epsilon, mu)

# ----------------------------- Experiment Setup ----------------------------- #
N_eps = 200
Considered_ns = [1000, 10000, 100000]
all_eps = np.linspace(1e-3, 1 - 1e-3, N_eps)

# Sweep over different null and alternative Bernoulli parameters
for gamma4null in [0.1, 0.3, 0.5, 0.7]:
    for mu4alternative in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if mu4alternative <= gamma4null:
            continue  # Skip non-identifiable cases
        else:
            print(f"Plot for gamma {gamma4null} and mu {mu4alternative}...")
            limit_eps = []
            results = []

            # Sweep over sample sizes
            for n in Considered_ns:
                estimated_eps = []

                for eps in all_eps:
                    # Step 1: Generate data
                    data = generate_mixture_data(n, eps, mu4alternative, gamma4null)

                    # Step 2: Estimate epsilon via MLE
                    est_eps, _ = estimate_eps_mu(data, null_gamma=gamma4null)
                    estimated_eps.append(est_eps)

                    # Step 3: Compute theoretical limit (only once for largest n)
                    if n == Considered_ns[-1]:
                        true_mean = (1 - eps) * gamma4null + eps * mu4alternative

                        def f(x):
                            # Target: match deviation from null mean in transformed domain
                            return np.abs(x**3 * np.sqrt(x**2 + gamma4null) - max(true_mean - gamma4null, 0))

                        # Use eps itself as init if small, otherwise 0.1
                        initial_eps = eps if eps <= 0.1 else 0.1
                        y = minimize(f, initial_eps, bounds=[[0, 1]]).x
                        l_eps = y * np.sqrt(y**2 + gamma4null)
                        limit_eps.append(l_eps)

                results.append(estimated_eps)

            # ---------------------------- Plotting ---------------------------- #
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

            # Plot estimated curves for each sample size
            for i, n in enumerate(Considered_ns):
                ax.plot(all_eps, results[i],
                        label=fr"MLE with $n={n}$", linestyle="dotted")

            # Plot ground truth and theoretical bound
            ax.plot(all_eps, all_eps, color="black", linestyle="dotted", label="True fraction")
            ax.plot(all_eps, limit_eps, color='red', linestyle='--', label="Limit fraction")

            ax.legend(loc='upper left')
            plt.xlabel("True fraction")
            plt.ylabel("Estimated fraction")
            plt.tight_layout()
            plt.savefig(f"plot/GreenRed-gamma{gamma4null}-mu{mu4alternative}.pdf", dpi=300)
