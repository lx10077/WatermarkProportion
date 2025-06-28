#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless plotting
import matplotlib.pyplot as plt

# Set plot style and LaTeX rendering
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
from watermark_generation import generate_inverse_watermark_text

# --------------------------- Experiment Settings --------------------------- #
N_eps = 200                          # Number of epsilon points
N0 = 500                             # Estimator hyperparameter
n_sample, n_legnth = 2500, 400       # Number of samples and sequence length
K = 1000                             # Vocabulary size
c = 5                                # Gumbel top-k parameter
Key = 2846513                        # Seed for deterministic watermarking
fig_dir = "plot"
os.makedirs(fig_dir, exist_ok=True)

def run(Delta, use_iterative):
    print("Delta:", Delta)
    print("Use iterative solver:", use_iterative)

    # ------------------------ Generate or Load Data ------------------------ #
    saved_file = "simulation_data/"
    dataset_name = f"Inverse-K{K}N{n_sample}c{c}key{Key}T{n_legnth}Delta{Delta}"
    generate_data = not os.path.exists(saved_file + dataset_name + ".pkl")

    if generate_data:
        os.makedirs(os.path.dirname(saved_file + dataset_name + ".pkl"), exist_ok=True)
        print("Generating simulation data...")

        # Generate null data from Uniform[0,1]
        null_Y = np.random.uniform(size=(n_sample, n_legnth))

        # Generate inverse watermarked data
        alte_Y = []
        for _ in tqdm(range(n_sample)):
            _, _, selected_difs, _, _, _ = generate_inverse_watermark_text(
                np.random.randint(0, 100, size=c).tolist(), K, n_legnth, c, Delta, Key
            )
            alte_Y.append(selected_difs[-n_legnth:])
        alte_Y = np.array(alte_Y)

        dataset = {"null": null_Y.tolist(), "alte": alte_Y.tolist()}
        with open(saved_file + dataset_name + '.pkl', 'wb') as pickle_file:
            pickle.dump(dataset, pickle_file)
    else:
        print("Loading simulation data...")
        with open(saved_file + dataset_name + '.pkl', 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)
        null_Y = np.array(dataset["null"])
        alte_Y = np.array(dataset["alte"])

    # Transform alte_Y so its distribution also lies in [0, 1] and the null is U(0, 1)
    alte_Y = (1 + alte_Y)**2

    # ---------------------- Mixture Generation Function ---------------------- #
    def mix_matrices(X, Y, eps):
        """Mix `X` (watermarked) and `Y` (null) data row-wise with fraction `eps`."""
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

    # -------------------------- Plot CDF of Mixtures -------------------------- #
    cdf_dir = f"{fig_dir}/inverse_Delta{Delta}_cdf.pdf"
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

    # ---------------------------- Load Estimators ----------------------------- #
    from estimator import (
        inital_estimator,
        refined_estimator_constant_weight,
        refined_estimator_optimal_weight
    )

    # ----------------------- Evaluate Estimation Error ------------------------ #
    estimation_result_dir = f"fig_results/{dataset_name}-Iter{use_iterative}.pkl"
    results_dict = {}
    get_estimation_result = not os.path.exists(estimation_result_dir)
    os.makedirs(os.path.dirname(estimation_result_dir), exist_ok=True)
    all_eps = np.linspace(1e-3, 1 - 1e-3, N_eps)

    if get_estimation_result:
        inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst = [], [], []
        refine_estimate_constant_v1_lst, refine_estimate_constant_v2_lst, refine_estimate_constant_v3_lst = [], [], []
        optimal_estimate_lst, alternative_df = [], []

        for i, eps in tqdm(enumerate(all_eps)):
            mix_data = mix_matrices(alte_Y, null_Y, eps)

            # For debugging: track how often alte_Y ≤ eps
            alternative_df.append(np.mean(alte_Y <= eps))

            # Run all estimators
            inital_estimate1 = inital_estimator(mix_data, delta=1e-1)
            inital_estimate2 = inital_estimator(mix_data, delta=1e-2)
            inital_estimate3 = inital_estimator(mix_data, delta=1e-3)

            refine_estimate_constant_v1 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-1)
            refine_estimate_constant_v2 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-2)
            refine_estimate_constant_v3 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-3)

            optimal_estimate = refined_estimator_optimal_weight(mix_data, alte_Y, use_iterative, n_sample, n_legnth, N0)

            # Store all results
            inital_estimate1_lst.append(inital_estimate1)
            inital_estimate2_lst.append(inital_estimate2)
            inital_estimate3_lst.append(inital_estimate3)
            refine_estimate_constant_v1_lst.append(refine_estimate_constant_v1)
            refine_estimate_constant_v2_lst.append(refine_estimate_constant_v2)
            refine_estimate_constant_v3_lst.append(refine_estimate_constant_v3)
            optimal_estimate_lst.append(optimal_estimate)

        # Compute absolute errors
        def to_error(arr): return np.abs(np.array(arr) - all_eps)
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
        # Load cached results
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

    # ---------------------- Plot Estimation Error vs Eps ---------------------- #
    def select_the_best(lst1, lst2, lst3):
        """Select the list with the best (smallest) error most frequently."""
        stacked = np.stack([lst1, lst2, lst3])
        min_values = np.min(stacked, axis=0)
        counts = [(stacked[i] == min_values).sum() for i in range(3)]
        return stacked[np.argmax(counts)]

    plt.figure(figsize=[8, 6])
    y = select_the_best(inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst)
    plt.plot(all_eps, inital_estimate2_lst, label=r"\textsf{INI} ($\widehat{\varepsilon}_{\mathrm{ini}}$)", linestyle="dotted")
    plt.plot(all_eps, refine_estimate_constant_v1_lst, label=r"\textsf{IND} ($\widehat{\varepsilon}_{\mathrm{rfn}}$)", linestyle="--")
    plt.plot(all_eps, optimal_estimate_lst, label=r"\textsf{OPT} ($\widehat{\varepsilon}_{\mathrm{opt}}$)", color="red", linestyle="--")
    plt.legend(loc='upper left')
    plt.ylabel("Mean absolute error")
    plt.xlabel("True fraction")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/finalfig-" + dataset_name + f"-Iter{use_iterative}.pdf", dpi=300)

    # ------------------------- Output LaTeX Table Row ------------------------- #
    def to_latex_stat(arr):
        """Return mean(std) scaled by 1e4 as integer tuple."""
        mean = int(round(np.mean(arr), 4) * 1e4)
        std = int(round(np.std(arr), 4) * 1e4)
        return f"{mean}({std})"

    a1 = to_latex_stat(y)
    a2 = to_latex_stat(refine_estimate_constant_v1_lst)
    a3 = to_latex_stat(optimal_estimate_lst)
    output_str = f"{Delta} & {a1} & {a2} &" + r"\textbf{" + a3 + "} \n"
    print(output_str)


# ----------------------------- Run Main Loop ----------------------------- #
if __name__ == "__main__":
    for Delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        Delta = str(Delta) + "+"
        for use_iterative in [False]:
            run(Delta, use_iterative)
