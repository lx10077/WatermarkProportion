#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib (for saving figures)
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'lines.linewidth': 2
})
from localization import OpenaiAligator
import argparse

# ======================= Argument Parsing ============================
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="Gumbel", type=str)  # Watermark method to use
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)  # HuggingFace model
parser.add_argument('--seed', default=15485863, type=int)  # Random seed for reproducibility
parser.add_argument('--c', default=4, type=int)  # Window size for watermark seeding
parser.add_argument('--m', default=500, type=int)  # Number of new tokens to generate
parser.add_argument('--T', default=500, type=int)  # Number of total prompts
parser.add_argument('--temp', default=1, type=float)  # Sampling temperature
parser.add_argument('--seed_way', default="noncomm_prf", type=str)  # PRF seeding scheme

args = parser.parse_args()
print(args)

# Histogram bin count used for density estimation
N0 = 500
use_iterative = False

# List of true corruption proportions (epsilon values)
all_eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

# Thresholds for different watermarking in WPL
threshold_Inverse = [0.73**2, 0.75**2, 0.77**2]
threshold_Gumbel = [1.3, 1.5, 1.7]

# Compute baseline MAE using the OpenaiAligator detector with multiple thresholds
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


# ==============================
# Main evaluation loop and plotting
# ==============================

# Import estimators
from estimator import (
    inital_estimator,
    refined_estimator_constant_weight,
    refined_estimator_optimal_weight
)

def select_the_best(lst1, lst2, lst3):
    # Stack the arrays for comparison
    stacked = np.stack([lst1, lst2, lst3])
    # Determine the smallest value at each position
    min_values = np.min(stacked, axis=0)
    # Count how many times each array achieves the minimum
    counts = [(stacked[i] == min_values).sum() for i in range(3)]
    # Identify the array with the most minimum entries
    selected_index = np.argmax(counts)
    selected_array = stacked[selected_index]
    return selected_array

def plot4(watermark, model_name, temp):
    size = "1p3B" if model_name == "facebook/opt-1.3b" else model_name.split("/")[-1]

    # Compute file naming and load estimation results if exist
    estimation_result_dir = f"fig_data/c4-mixture-{watermark}-{size}-N0{N0}-temp{temp}-Iter{use_iterative}.pkl"
    results_dict = dict()
    get_estimation_result = os.path.exists(estimation_result_dir)
    os.makedirs(os.path.dirname(estimation_result_dir), exist_ok=True)

    # Load data and compute all estimators
    mixed_Y, whether_watermark, top_prob = dict(), dict(), dict()
    true_eps = []
    for eps in all_eps:
        file_dir = f"text_data/{size}-{watermark}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-eps{eps}.pkl"
        with open(file_dir, 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)
        Y = dataset["watermark"]["Ys"].numpy()
        if watermark == "Inverse":
            Y = (1+Y)**2
        mixed_Y[eps] = Y
        used_watermark = dataset["watermark"]["where_watermark"].numpy()
        whether_watermark[eps] = used_watermark
        true_eps.append(np.mean(used_watermark))
        top_prob[eps] = dataset["watermark"]["top_probs"].numpy()
    true_eps = np.array(true_eps)

    def gather_watermarked_Y(mixed_Y, whether_watermark):
        arrays = []
        for eps in all_eps:
            watermarked_Y = mixed_Y[eps][whether_watermark[eps]]
            arrays.append(watermarked_Y)
        pooled = np.concatenate(arrays)
        return pooled.reshape(-1)

    alte_Y = gather_watermarked_Y(mixed_Y, whether_watermark)

    if get_estimation_result:
        with open(estimation_result_dir, 'rb') as f:
            results_dict = pickle.load(f)

        # Load previously computed arrays
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
        # Initialize estimator result lists
        inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst = [], [], []
        refine_estimate_constant_v1_lst, refine_estimate_constant_v2_lst, refine_estimate_constant_v3_lst = [], [], []
        optimal_estimate_lst, alternative_df, baseline_estimate_lst = [], [], []

        for _, eps in tqdm(enumerate(all_eps)):
            mix_data = mixed_Y[eps]

            alternative_df.append(np.mean(alte_Y <= eps))
            baseline_estimate = compute_baseline_MAE(mix_data, watermark) 

            inital_estimate1 = inital_estimator(mix_data, delta=1e-1)
            inital_estimate2 = inital_estimator(mix_data, delta=1e-2)
            inital_estimate3 = inital_estimator(mix_data, delta=1e-3)

            refine_estimate_constant_v1 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-1)
            refine_estimate_constant_v2 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-2)
            refine_estimate_constant_v3 = refined_estimator_constant_weight(mix_data, alte_Y, delta=1e-3)

            optimal_estimate = refined_estimator_optimal_weight(mix_data, alte_Y, use_iterative, N0)

            baseline_estimate_lst.append(baseline_estimate)
            inital_estimate1_lst.append(inital_estimate1)
            inital_estimate2_lst.append(inital_estimate2)
            inital_estimate3_lst.append(inital_estimate3)
            refine_estimate_constant_v1_lst.append(refine_estimate_constant_v1)
            refine_estimate_constant_v2_lst.append(refine_estimate_constant_v2)
            refine_estimate_constant_v3_lst.append(refine_estimate_constant_v3)
            optimal_estimate_lst.append(optimal_estimate)

        # Convert to np.array and compute absolute error
        baseline_estimate_lst = np.array(baseline_estimate_lst)
        inital_estimate1_lst = np.abs(np.array(inital_estimate1_lst) - true_eps)
        inital_estimate2_lst = np.abs(np.array(inital_estimate2_lst) - true_eps)
        inital_estimate3_lst = np.abs(np.array(inital_estimate3_lst) - true_eps)
        refine_estimate_constant_v1_lst = np.abs(np.array(refine_estimate_constant_v1_lst) - true_eps)
        refine_estimate_constant_v2_lst = np.abs(np.array(refine_estimate_constant_v2_lst) - true_eps)
        refine_estimate_constant_v3_lst = np.abs(np.array(refine_estimate_constant_v3_lst) - true_eps)
        optimal_estimate_lst = np.abs(np.array(optimal_estimate_lst) - true_eps)

        results_dict = {
            "alternatve_CDF": np.array(alternative_df),
            "baseline_estimate": baseline_estimate_lst,
            "inital_estimate1": inital_estimate1_lst,
            "inital_estimate2": inital_estimate2_lst,
            "inital_estimate3": inital_estimate3_lst,
            "refine_estimate_constant_v1": refine_estimate_constant_v1_lst,
            "refine_estimate_constant_v2": refine_estimate_constant_v2_lst,
            "refine_estimate_constant_v3": refine_estimate_constant_v3_lst,
            "optimal_estimate_lst": optimal_estimate_lst,
        }

        with open(estimation_result_dir, 'wb') as f:
            pickle.dump(results_dict, f)

    # Plot the results
    plt.figure(figsize=[8, 6])
    y = select_the_best(inital_estimate1_lst, inital_estimate2_lst, inital_estimate3_lst)
    plt.plot(true_eps, y, label = r"\textsf{INI} ($\widehat{\varepsilon}_{\mathrm{ini}}$)", linestyle="dotted")
    plt.plot(true_eps, refine_estimate_constant_v1_lst, label = r"\textsf{IND} ($\widehat{\varepsilon}_{\mathrm{rfn}}$)", linestyle="-.")
    plt.plot(true_eps, optimal_estimate_lst, label = r"\textsf{OPT} ($\widehat{\varepsilon}_{\mathrm{opt}}$)", color="red", linestyle="--")
    baseline_estimate_lst = np.min(np.abs(baseline_estimate_lst[:,:3]-true_eps[:,np.newaxis]), axis=1)
    plt.plot(true_eps, baseline_estimate_lst, label = r"\textsf{WPL}", color="black", linestyle=":")

    plt.ylim(top=0.3)
    plt.legend(loc='upper left', fontsize=20)
    plt.ylabel("Mean absolute error")
    plt.xlabel("True proportion")
    plt.tight_layout()
    plt.savefig(f"figs/c4-miture-{size}-{watermark}-N0{N0}-temp{temp}-Iter{use_iterative}.pdf", dpi=300)

    # Print formatted table result for LaTeX
    effective_number = 3
    a1 = int(round(np.mean(y), effective_number) * 10**effective_number)
    b1 = int(round(np.std(y), effective_number) * 10**effective_number)
    a2 = int(round(np.mean(refine_estimate_constant_v1_lst), effective_number) * 10**effective_number)
    b2 = int(round(np.std(refine_estimate_constant_v1_lst), effective_number) * 10**effective_number)
    a3 = int(round(np.mean(optimal_estimate_lst), effective_number) * 10**effective_number)
    b3 = int(round(np.std(optimal_estimate_lst), effective_number) * 10**effective_number)
    a4 = int(round(np.mean(baseline_estimate_lst), effective_number) * 10**effective_number)
    b4 = int(round(np.std(baseline_estimate_lst), effective_number) * 10**effective_number)

    output_str = f"{watermark},{size},{temp} & {a4}({b4}) & {a1}({b1}) & {a2}({b2}) &" + r"\textbf{" + f"{a3}({b3})" + "} \\"
    print(output_str)

if __name__ == "__main__":
    plot4(args.method, args.model, args.temp)
