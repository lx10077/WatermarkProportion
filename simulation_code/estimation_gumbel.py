import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# Set plot style and LaTeX formatting
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'lines.linewidth': 2,
})

from watermark_generation import generate_gumbel_watermark_text

# ----------------------------- Hyperparameters ----------------------------- #
N_eps = 200  # Number of eps values to try
N0 = 500     # Parameter used in the optimal estimator
n_sample, n_legnth = 2500, 400  # Dataset size and sequence length
K = 1000     # Vocabulary size for watermark generation
c = 5        # Gumbel top-k parameter
Key = 2846513  # Random seed for watermark generation
fig_dir = "plot"
os.makedirs(fig_dir, exist_ok=True)


def run(Delta, use_iterative):
    print("Delta:", Delta)
    print("Use iterative solver:", use_iterative)

    # ------------------- Generate or Load Simulated Data ------------------- #
    saved_file = "simulation_data/"
    dataset_name = f"Gumbel-K{K}N{n_sample}c{c}key{Key}T{n_legnth}Delta{Delta}"
    generate_data = not os.path.exists(saved_file + dataset_name + ".pkl")

    if generate_data:
        os.makedirs(os.path.dirname(saved_file + dataset_name + ".pkl"), exist_ok=True)
        print("Generating simulation data...")
        
        # Generate null data from uniform distribution
        null_Y = np.random.uniform(size=(n_sample, n_legnth))
        
        # Generate watermarked data using the Gumbel-max method
        alte_Y = []
        for _ in tqdm(range(n_sample)):
            _, selected_xis, _ = generate_gumbel_watermark_text(
                np.random.randint(0, 100, size=c).tolist(), K, n_legnth, c, Delta, Key
            )
            alte_Y.append(selected_xis[-n_legnth:])
        alte_Y = np.array(alte_Y)

        # Save generated data
        dataset = {"null": null_Y.tolist(), "alte": alte_Y.tolist()}
        with open(saved_file + dataset_name + '.pkl', 'wb') as pickle_file:
            pickle.dump(dataset, pickle_file)

    else:
        print("Loading simulation data...")
        with open(saved_file + dataset_name + '.pkl', 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)
        null_Y = np.array(dataset["null"])
        alte_Y = np.array(dataset["alte"])

    # ------------------------ Mix Watermarked and Null ---------------------- #
    def mix_matrices(X, Y, eps):
        """Mix `X` (watermarked) and `Y` (null) row-wise according to fraction `eps`."""
        if eps == 0:
            return Y
        elif eps == 1:
            return X
        elif not (0 <= eps <= 1):
            raise ValueError("eps must be between 0 and 1.")
        
        num_rows, num_cols = X.shape
        num_from_X = int(np.floor(eps * num_cols))
        mixed = np.zeros_like(X)
        
        for i in range(num_rows):
            indices_X = np.random.choice(num_cols, num_from_X, replace=False)
            indices_Y = np.setdiff1d(np.arange(num_cols), indices_X)
            mixed[i, indices_X] = X[i, indices_X]
            mixed[i, indices_Y] = Y[i, indices_Y]
        
        return mixed

    # --------------------- Plot CDFs of Mixed Distributions ----------------- #
    cdf_dir = f"{fig_dir}/gumbel_Delta{Delta}_cdf.pdf"
    plot_cdf = not os.path.exists(cdf_dir)
    os.makedirs(os.path.dirname(cdf_dir), exist_ok=True)

    if plot_cdf:
        plt.figure(figsize=[8, 6])
        print("Start plotting CDF...")
        for eps in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
            x = np.sort(mix_matrices(alte_Y, null_Y, eps).reshape(-1))
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, linestyle='--', label=fr"$\varepsilon={eps}$")
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='dotted', color="black", label="Uniform(0, 1)")
        plt.xlabel(r'$Y$')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cdf_dir, dpi=300)
        print("Finish plotting...")

    # ----------------------------- Estimation Code -------------------------- #
    from estimator import (
        inital_estimator,
        refined_estimator_constant_weight,
        refined_estimator_optimal_weight
    )

    # ------------------ Evaluate Estimation Error over Eps ------------------ #
    estimation_result_dir = f"fig_results/{dataset_name}-Iter{use_iterative}.pkl"
    results_dict = {}
    get_estimation_result = not os.path.exists(estimation_result_dir)
    os.makedirs(os.path.dirname(estimation_result_dir), exist_ok=True)
    all_eps = np.linspace(1e-3, 1 - 1e-3, N_eps)

    if get_estimation_result:
        # Store results for different estimator configurations
        inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst = [], [], []
        refine_estimate_constant_v1_lst, refine_estimate_constant_v2_lst, refine_estimate_constant_v3_lst = [], [], []
        optimal_estimate_lst, alternative_df = [], []

        for i, eps in tqdm(enumerate(all_eps)):
            mix_data = mix_matrices(alte_Y, null_Y, eps)
            alternative_df.append(np.mean(alte_Y <= eps))

            # Initial estimators with varying delta
            inital_estimate1_lst.append(inital_estimator(mix_data, delta=1e-1))
            inital_estimate2_lst.append(inital_estimator(mix_data, delta=1e-2))
            inital_estimate3_lst.append(inital_estimator(mix_data, delta=1e-3))

            # Refined estimators with constant weights
            refine_estimate_constant_v1_lst.append(refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-1))
            refine_estimate_constant_v2_lst.append(refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-2))
            refine_estimate_constant_v3_lst.append(refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-3))

            # Optimal estimator using optimal weights
            optimal_estimate_lst.append(refined_estimator_optimal_weight(
                mix_data, alte_Y, use_iterative, n_sample, n_legnth, N0
            ))

        # Compute estimation error (MAE) against true eps
        def to_error(estimates): return np.abs(np.array(estimates) - all_eps)
        results_dict["alternatve_CDF"] = np.array(alternative_df)
        results_dict["inital_estimate1"] = to_error(inital_estimate1_lst)
        results_dict["inital_estimate2"] = to_error(inital_estimate2_lst)
        results_dict["inital_estimate3"] = to_error(inital_estimate3_lst)
        results_dict["refine_estimate_constant_v1"] = to_error(refine_estimate_constant_v1_lst)
        results_dict["refine_estimate_constant_v2"] = to_error(refine_estimate_constant_v2_lst)
        results_dict["refine_estimate_constant_v3"] = to_error(refine_estimate_constant_v3_lst)
        results_dict["optimal_estimate_lst"] = to_error(optimal_estimate_lst)

        # Save results
        with open(estimation_result_dir, 'wb') as f:
            pickle.dump(results_dict, f)

    else:
        # Load previously saved estimation errors
        with open(estimation_result_dir, 'rb') as f:
            results_dict = pickle.load(f)

        alternative_df = results_dict["alternatve_CDF"]
        inital_estimate1_lst = results_dict["inital_estimate1"]
        inital_estimate2_lst = results_dict["inital_estimate2"]
        inital_estimate3_lst = results_dict["inital_estimate3"]
        refine_estimate_constant_v1_lst = results_dict["refine_estimate_constant_v1"]
        refine_estimate_constant_v2_lst = results_dict["refine_estimate_constant_v2"]
        refine_estimate_constant_v3_lst = results_dict["refine_estimate_constant_v3"]
        optimal_estimate_lst = results_dict["optimal_estimate_lst"]

    # ---------------------- Plot Estimation Error vs Eps -------------------- #
    def select_the_best(lst1, lst2, lst3):
        """Select the best estimator (least error) among three lists."""
        stacked = np.stack([lst1, lst2, lst3])
        min_values = np.min(stacked, axis=0)
        counts = [(stacked[i] == min_values).sum() for i in range(3)]
        selected_index = np.argmax(counts)
        return stacked[selected_index]

    plt.figure(figsize=[8, 6])
    y = select_the_best(inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst)
    plt.plot(all_eps, y, label=r"\textsf{INI} ($\widehat{\varepsilon}_{\mathrm{ini}}$)", linestyle="dotted")
    plt.plot(all_eps, refine_estimate_constant_v1_lst, label=r"\textsf{IND} ($\widehat{\varepsilon}_{\mathrm{rfn}}$)", linestyle="--")
    plt.plot(all_eps, optimal_estimate_lst, label=r"\textsf{OPT} ($\widehat{\varepsilon}_{\mathrm{opt}}$)", color="red", linestyle="--")
    plt.legend(loc='upper left')
    plt.ylabel("Mean absolute error")
    plt.xlabel("True fraction")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/finalfig-" + dataset_name + f"-Iter{use_iterative}.pdf", dpi=300)

    # -------------------------- LaTeX Table Output -------------------------- #
    def to_latex_entry(arr):
        """Return rounded mean (std) as LaTeX-friendly string."""
        a = int(round(np.mean(arr), 4) * 1e4)
        b = int(round(np.std(arr), 4) * 1e4)
        return f"{a}({b})"

    output_str = f"{Delta} & {to_latex_entry(y)} & {to_latex_entry(refine_estimate_constant_v1_lst)} &" + \
                 r"\textbf{" + to_latex_entry(optimal_estimate_lst) + "} \n"
    print(output_str)


if __name__ == "__main__":
    # Sweep over different Delta values and run estimation once per setting
    for Delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        Delta = str(Delta) + "+"
        for use_iterative in [False]:
            run(Delta, use_iterative)
