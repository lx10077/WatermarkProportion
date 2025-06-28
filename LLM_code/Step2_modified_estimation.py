
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import standard libraries and third-party packages
import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Set global plotting parameters
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'lines.linewidth': 2
})

from localization import OpenaiAligator  # Custom class for detecting watermark regions

# General configuration
N0 = 500
use_iterative = False
c = 4
m = 500

# Different corruption budgets to evaluate robustness
modify_budgets = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]

# Threshold values for different watermarking methods
threshold_Inverse = [0.73**2, 0.75**2, 0.77**2]
threshold_Gumbel = [1.3, 1.5, 1.7]

# Compute baseline estimation using Aligator
def compute_baseline_MAE(mixed_Ys, watermark):
    if watermark == "Gumbel":
        thresholds = threshold_Gumbel
        score_Ys = -np.log(1 - mixed_Ys)
    else:
        thresholds = threshold_Inverse
        score_Ys = mixed_Ys

    results = []
    for each_threhold in thresholds:
        localizator = OpenaiAligator(each_threhold)
        estimate_eps = localizator.compute_fraction_whole(score_Ys)
        results.append(estimate_eps)
    return np.array(results)

# Import estimators
from estimator import (
    inital_estimator,
    refined_estimator_constant_weight,
    refined_estimator_optimal_weight
)

# Main function to evaluate different estimators
def plot4(task, watermark, model_name, temp):
    mixed_Y = dict()
    whether_watermark = dict()

    if temp == "1" and model_name != "1p3B":
        temp = "1.0"  # Normalize temp string

    true_eps = []

    # Load corrupted dataset for different modification budgets
    for budget in modify_budgets:
        exp_name = f"text_data/{model_name}-{watermark}-c4-m500-T500-noncomm_prf-15485863-temp{temp}-{task}{budget}.pkl"
        if not os.path.exists(exp_name):
            raise FileNotFoundError(f"No such file: {exp_name}")
        with open(exp_name, "rb") as f:
            results = pickle.load(f)

        modify_mask = results["where_modify"].numpy()
        corrupted_Y = results["Ys"]

        # For Inverse watermark, adjust scores
        if watermark == "Inverse":
            corrupted_Y = (1 + corrupted_Y) ** 2

        mixed_Y[budget] = corrupted_Y

        # Determine where the watermark is still present
        whether_watermark[budget] = ~modify_mask.squeeze()
        true_ep = np.mean(whether_watermark[budget])
        true_eps.append(true_ep)

    true_eps = np.array(true_eps)

    # Optionally overwrite true_eps with pre-computed values
    if task != "sub":
        true_eps = np.array([0.858836, 0.773324, 0.695484, 0.625032, 0.557948,
                             0.498844, 0.444976, 0.39272, 0.34916, 0.308352, 0.271004])

    # Load uncorrupted baseline sample for comparison
    raw_exp_name = f"text_data/{model_name}-{watermark}-c4-m500-T500-noncomm_prf-15485863-temp{temp}-raw.pkl"
    if not os.path.exists(raw_exp_name):
        raise FileNotFoundError(f"No such file: {raw_exp_name}")
    with open(raw_exp_name, "rb") as f:
        raw_results = pickle.load(f)

    alte_Y = raw_results["watermark"]["Ys"].numpy().reshape(-1)
    if watermark == "Inverse":
        alte_Y = (1 + alte_Y) ** 2

    # Path to store result for current config
    estimation_result_dir = f"fig_data/c4-{task}-{model_name}-{watermark}-{model_name}-N0{N0}-temp{temp}-Iter{use_iterative}.pkl"
    os.makedirs(os.path.dirname(estimation_result_dir), exist_ok=True)

    results_dict = dict()
    get_estimation_result = os.path.exists(estimation_result_dir)

    # Load previous results if available
    if get_estimation_result:
        with open(estimation_result_dir, 'rb') as f:
            results_dict = pickle.load(f)

        alternative_df = results_dict["alternatve_CDF"]
        baseline_estimate_lst = results_dict["baseline_estimate"]
        inital_estimate1_lst = results_dict["inital_estimate1"]
        inital_estimate2_lst = results_dict["inital_estimate2"]
        inital_estimate3_lst = results_dict["inital_estimate3"]
        refine_estimate_constant_v1_lst = results_dict["refine_estimate_constant_v1"]
        refine_estimate_constant_v2_lst = results_dict["refine_estimate_constant_v2"]
        refine_estimate_constant_v3_lst = results_dict["refine_estimate_constant_v3"]
        optimal_estimate_lst = results_dict["optimal_estimate_lst"]

    else:
        # Otherwise compute all estimates from scratch
        alternative_df = []
        baseline_estimate_lst = []
        inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst = [], [], []
        refine_estimate_constant_v1_lst, refine_estimate_constant_v2_lst, refine_estimate_constant_v3_lst = [], [], []
        optimal_estimate_lst = []

        for i, budget in tqdm(enumerate(modify_budgets)):
            mix_data = mixed_Y[budget]

            alternative_df.append(np.mean(alte_Y <= true_eps[i]))                    
            baseline_estimate = compute_baseline_MAE(mix_data, watermark) 

            inital_estimate1 = inital_estimator(mix_data, delta=1e-1)
            inital_estimate2 = inital_estimator(mix_data, delta=1e-2)
            inital_estimate3 = inital_estimator(mix_data, delta=1e-3)

            refine_estimate_constant_v1 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-1)
            refine_estimate_constant_v2 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-2)
            refine_estimate_constant_v3 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-3)

            optimal_estimate = refined_estimator_optimal_weight(mix_data, alte_Y)

            baseline_estimate_lst.append(baseline_estimate)
            inital_estimate1_lst.append(inital_estimate1)
            inital_estimate2_lst.append(inital_estimate2)
            inital_estimate3_lst.append(inital_estimate3)
            refine_estimate_constant_v1_lst.append(refine_estimate_constant_v1)
            refine_estimate_constant_v2_lst.append(refine_estimate_constant_v2)
            refine_estimate_constant_v3_lst.append(refine_estimate_constant_v3)
            optimal_estimate_lst.append(optimal_estimate)

        # Compute absolute errors
        baseline_estimate_lst = np.array(baseline_estimate_lst)
        inital_estimate1_lst = np.abs(np.array(inital_estimate1_lst) - true_eps)
        inital_estimate2_lst = np.abs(np.array(inital_estimate2_lst) - true_eps)
        inital_estimate3_lst = np.abs(np.array(inital_estimate3_lst) - true_eps)
        refine_estimate_constant_v1_lst = np.abs(np.array(refine_estimate_constant_v1_lst) - true_eps)
        refine_estimate_constant_v2_lst = np.abs(np.array(refine_estimate_constant_v2_lst) - true_eps)
        refine_estimate_constant_v3_lst = np.abs(np.array(refine_estimate_constant_v3_lst) - true_eps)
        optimal_estimate_lst = np.abs(np.array(optimal_estimate_lst) - true_eps)

        # Save all computed results
        results_dict.update({
            "alternatve_CDF": np.array(alternative_df),
            "baseline_estimate": baseline_estimate_lst,
            "inital_estimate1": inital_estimate1_lst,
            "inital_estimate2": inital_estimate2_lst,
            "inital_estimate3": inital_estimate3_lst,
            "refine_estimate_constant_v1": refine_estimate_constant_v1_lst,
            "refine_estimate_constant_v2": refine_estimate_constant_v2_lst,
            "refine_estimate_constant_v3": refine_estimate_constant_v3_lst,
            "optimal_estimate_lst": optimal_estimate_lst
        })

        with open(estimation_result_dir, 'wb') as f:
            pickle.dump(results_dict, f)

    # Helper: select the best-performing initial estimator
    def select_the_best(lst1, lst2, lst3):
        stacked = np.stack([lst1, lst2, lst3])
        min_values = np.min(stacked, axis=0)
        counts = [(stacked[i] == min_values).sum() for i in range(3)]
        selected_index = np.argmax(counts)
        return stacked[selected_index]

    # Plotting
    plt.figure(figsize=[8, 6])
    y = select_the_best(inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst)
    plt.plot(true_eps, y, label = r"\textsf{INI} ($\widehat{\varepsilon}_{\mathrm{ini}}$)", linestyle="dotted")
    plt.plot(true_eps, refine_estimate_constant_v1_lst, label = r"\textsf{IND} ($\widehat{\varepsilon}_{\mathrm{rfn}}$)", linestyle="-.")
    plt.plot(true_eps, optimal_estimate_lst, label = r"\textsf{OPT} ($\widehat{\varepsilon}_{\mathrm{opt}}$)", color="red", linestyle="--")
    baseline_estimate_lst = np.min(np.abs(baseline_estimate_lst - true_eps[:, np.newaxis]), axis=1)
    plt.plot(true_eps, baseline_estimate_lst, label = r"\textsf{WPL}", color="black", linestyle=":")
    plt.legend(loc='upper left', fontsize=20)
    plt.ylabel("Mean absolute error")
    plt.xlabel("True proportion")
    plt.tight_layout()
    plt.savefig(f"figs/c4-{task}-{model_name}-{watermark}-N0{N0}-temp{temp}-Iter{use_iterative}.pdf", dpi=300)

    # Print summary in LaTeX format
    effective_number = 3
    a1 = int(round(np.mean(y), effective_number) * 10**effective_number)
    b1 = int(round(np.std(y), effective_number) * 10**effective_number)
    a2 = int(round(np.mean(refine_estimate_constant_v1_lst), effective_number) * 10**effective_number)
    b2 = int(round(np.std(refine_estimate_constant_v1_lst), effective_number) * 10**effective_number)
    a3 = int(round(np.mean(optimal_estimate_lst), effective_number) * 10**effective_number)
    b3 = int(round(np.std(optimal_estimate_lst), effective_number) * 10**effective_number)
    a4 = int(round(np.mean(baseline_estimate_lst), effective_number) * 10**effective_number)
    b4 = int(round(np.std(baseline_estimate_lst), effective_number) * 10**effective_number)

    output_str = f"{watermark},{model_name},{temp}, {task} & {a4}({b4}) & {a1}({b1}) & {a2}({b2}) &" + r"\textbf{" + f"{a3}({b3})" + "} \\"
    print(output_str)
    print()
    return

if __name__ == "__main__":
    for task in ["sub", "del", "ins"]:
        plot4(task, "Gumbel", "1p3B", "1")
