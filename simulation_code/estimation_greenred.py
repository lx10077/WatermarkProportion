#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'lines.linewidth': 2,
})
from scipy.optimize import minimize
from watermark_generation import generate_greenred_watermark_text

# Global config
N_eps = 200
N0 = 500
n_sample, n_legnth = 2500, 400
K = 1000
c = 5
Key = 2846513
fig_dir = "plot"


def run(Delta, use_iterative, gamma, delta):
    print("Delta:", Delta)
    print("Use iterative solver:", use_iterative)

    ### Step 1: Load or generate simulation data
    saved_file = "simulation_data/"
    dataset_name = f"Greenred(ga{gamma},de{delta})-K{K}N{n_sample}c{c}key{Key}T{n_legnth}Delta{Delta}"
    data_path = saved_file + dataset_name + ".pkl"
    generate_data = not os.path.exists(data_path)

    if generate_data:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print("Generating simulation data...")
        null_Y = np.random.binomial(n=1, p=gamma, size=(n_sample, n_legnth))
        alte_Y = []
        for _ in tqdm(range(n_sample)):
            inputs, selected_xis, _ = generate_greenred_watermark_text(
                np.random.randint(0, 100, size=c).tolist(), K, n_legnth, c, Delta, Key, gamma, delta
            )
            alte_Y.append(selected_xis[-n_legnth:])
        alte_Y = np.array(alte_Y)

        dataset = {"null": null_Y.tolist(), "alte": alte_Y.tolist()}
        with open(data_path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        print("Loading simulation data...")
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        null_Y = np.array(dataset["null"])
        alte_Y = np.array(dataset["alte"])

    def mix_matrices(X, Y, eps):
        if eps == 0:
            return Y
        elif eps == 1:
            return X
        else:
            if not (0 <= eps <= 1):
                raise ValueError("eps must be between 0 and 1.")
            num_rows, num_cols = X.shape
            num_from_X = int(np.floor(eps * num_cols))
            mixed = np.zeros_like(X)
            for i in range(num_rows):
                idx_X = np.random.choice(num_cols, num_from_X, replace=False)
                idx_Y = np.setdiff1d(np.arange(num_cols), idx_X)
                mixed[i, idx_X] = X[i, idx_X]
                mixed[i, idx_Y] = Y[i, idx_Y]
            return mixed

    ### Step 2: Plot CDFs
    cdf_path = f"{fig_dir}/greenred(ga{gamma},de{delta})_Delta{Delta}_cdf.pdf"
    if not os.path.exists(cdf_path):
        os.makedirs(os.path.dirname(cdf_path), exist_ok=True)
        print("Plotting CDFs...")
        plt.figure(figsize=[8, 6])
        for eps in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
            data = np.sort(mix_matrices(alte_Y, null_Y, eps).reshape(-1))
            cdf = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, cdf, linestyle='--', label=fr"$\varepsilon={eps}$")
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='dotted', color="black", label="Uniform(0, 1)")
        plt.xlabel(r'$Y$')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cdf_path, dpi=300)
        print("CDF plot saved.")

    ### Step 3: Estimators
    def inital_estimator(obs_Y, delta=0.1):
        return np.mean(np.clip(1 - np.mean(obs_Y <= delta, axis=1) / delta, 0., 1.))

    def refined_estimator_constant_weight(obs_Y, wm_Y, delta=0.1):
        numer = np.mean(obs_Y <= delta, axis=1) - delta + 1e-6
        denom = np.mean(wm_Y <= delta, axis=1) - delta + 1e-6
        return np.mean(np.clip(numer / denom, 0., 1.))

    def histgram_density(data, N=500):
        data = data.reshape(-1)
        counts, bin_edges = np.histogram(data, bins=N, range=(0, 1), density=True)
        return bin_edges, counts

    def evaluate_density(values, bin_edges, density):
        indices = np.searchsorted(bin_edges, values.reshape(-1), side="right") - 1
        indices = np.clip(indices, 0, len(density) - 1)
        return density[indices]

    def refined_estimator_optimal_weight(obs_Y, cdf, use_iterative=use_iterative, small=1e-6):
        bin_edges, dens = histgram_density(cdf, N=N0)
        unif_vals = np.random.uniform(0, 1, n_sample * n_legnth)
        f0 = evaluate_density(unif_vals, bin_edges, dens)
        fp = evaluate_density(cdf, bin_edges, dens)
        fy = evaluate_density(obs_Y, bin_edges, dens)

        def iterate(eps):
            E0 = np.mean((1 - f0 + small) / ((1 - eps) + eps * f0 + small))
            Ep = np.mean((1 - fp + small) / ((1 - eps) + eps * fp + small))
            EY = np.mean((1 - fy + small) / ((1 - eps) + eps * fy + small))
            return np.clip((E0 - EY) / (E0 - Ep), 1e-3, 1)

        if use_iterative:
            eps = 0.5
            for _ in range(1000):
                eps = iterate(eps)
        else:
            def loss_fn(eps):
                return abs(eps - iterate(eps))
            eps_init = 0.9
            for _ in range(20):
                eps_init = iterate(eps_init)
            res = minimize(loss_fn, eps_init, bounds=[(0, 1)])
            eps = res.x[0]
        return np.clip(eps, 0, 1)

    ### Step 4: Evaluate estimators
    results_path = f"fig_results/{dataset_name}-Iter{use_iterative}.pkl"
    get_estimation_result = not os.path.exists(results_path)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if get_estimation_result:
        
        all_eps = np.linspace(1e-3, 1 - 1e-3, N_eps)
        alt_cdfs = []
        ini1, ini2, ini3 = [], [], []
        rf1, rf2, rf3 = [], [], []
        opt = []

        for eps in tqdm(all_eps):
            mixed = mix_matrices(alte_Y, null_Y, eps)
            alt_cdfs.append(np.mean(alte_Y <= eps))

            ini1.append(inital_estimator(mixed, delta=1e-1))
            ini2.append(inital_estimator(mixed, delta=1e-2))
            ini3.append(inital_estimator(mixed, delta=1e-3))

            rf1.append(refined_estimator_constant_weight(mixed, alte_Y, delta=1e-1))
            rf2.append(refined_estimator_constant_weight(mixed, alte_Y, delta=1e-2))
            rf3.append(refined_estimator_constant_weight(mixed, alte_Y, delta=1e-3))

            opt.append(refined_estimator_optimal_weight(mixed, alte_Y))

        def mae(estimates):
            return np.abs(np.array(estimates) - all_eps)

        results_dict = {
            "alternatve_CDF": np.array(alt_cdfs),
            "inital_estimate1": mae(ini1),
            "inital_estimate2": mae(ini2),
            "inital_estimate3": mae(ini3),
            "refine_estimate_constant_v1": mae(rf1),
            "refine_estimate_constant_v2": mae(rf2),
            "refine_estimate_constant_v3": mae(rf3),
            "optimal_estimate_lst": mae(opt)
        }

        with open(results_path, 'wb') as f:
            pickle.dump(results_dict, f)
    else:
        with open(results_path, 'rb') as f:
            results_dict = pickle.load(f)

    ### Final comparison
    def select_best(lst1, lst2, lst3):
        stacked = np.stack([lst1, lst2, lst3])
        min_vals = np.min(stacked, axis=0)
        best_idx = np.argmax([(stacked[i] == min_vals).sum() for i in range(3)])
        return stacked[best_idx]

    plt.figure(figsize=[8, 6])
    best_ini = select_best(results_dict["inital_estimate1"], results_dict["inital_estimate2"], results_dict["inital_estimate3"])
    plt.plot(all_eps, best_ini, label=r"\textsf{INI} ($\widehat{\varepsilon}_{\mathrm{ini}}$)", linestyle="dotted")
    plt.plot(all_eps, results_dict["refine_estimate_constant_v1"], label=r"\textsf{IND} ($\widehat{\varepsilon}_{\mathrm{rfn}}$)", linestyle="--")
    plt.plot(all_eps, results_dict["optimal_estimate_lst"], label=r"\textsf{OPT} ($\widehat{\varepsilon}_{\mathrm{opt}}$)", color="red", linestyle="--")
    plt.legend(loc='upper left', fontsize=20)
    plt.ylabel("Mean absolute error")
    plt.xlabel("True proportion")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/finalfig-{dataset_name}-Iter{use_iterative}.pdf", dpi=300)


# Run the simulation
if __name__ == "__main__":
    for Delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        Delta = str(Delta) + "+"
        for use_iterative in [False]:
            run(Delta, use_iterative, gamma=0.5, delta=2)
