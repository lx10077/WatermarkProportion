# Import necessary libraries
from time import time
import os
import torch
import json
from generation import WatermarkGenerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
import numpy as np
import argparse

# -------------------------------
# Argument parsing for experiment settings
# -------------------------------
parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)  # Model to use
parser.add_argument('--seed', default=15485863, type=int)              # Random seed
parser.add_argument('--c', default=4, type=int)                        # Window size
parser.add_argument('--m', default=500, type=int)                      # Number of tokens to generate
parser.add_argument('--T', default=500, type=int)                      # Number of prompts
parser.add_argument('--seed_way', default="noncomm_prf", type=str)    # Watermark seeding scheme

parser.add_argument('--prompt_tokens', default=50, type=int)          # Number of tokens in the prompt
parser.add_argument('--buffer_tokens', default=20, type=int)          # Token buffer to leave room for new tokens
parser.add_argument('--max_seed', default=100000, type=int)

parser.add_argument('--norm', default=1, type=int)

parser.add_argument('--rt_translate', action='store_true')            # Real-time translation flag
parser.add_argument('--language', default="french", type=str)         # Target language for translation
parser.add_argument('--truncate_vocab', default=8, type=int)          # Vocab truncation from the end

args = parser.parse_args()
print(args)

# -------------------------------
# Main experiment loop
# -------------------------------

def run(method, temp):
    print(method, temp)

    # Set random seed for reproducibility
    t0 = time()
    torch.manual_seed(args.seed)

    print(f"Using {torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU.")

    # Load tokenizer and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", offload_folder="./offload_folder")
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    eff_vocab_size = vocab_size - args.truncate_vocab
    print(f'Loaded the model (t = {time()-t0} seconds)')

    # -------------------------------
    # Generation configuration
    # -------------------------------
    batch_size = 1 if method == "Inverse" else 10
    T = args.T
    n_batches = int(np.ceil(T / batch_size))
    new_tokens = args.m
    load_local_data = True
    buffer_tokens = args.buffer_tokens
    prompt_tokens = args.prompt_tokens

    # -------------------------------
    # Load dataset
    # -------------------------------
    if load_local_data:
        with open('c4/c4.json', 'r') as f:
            lines = f.readlines()
        ds_iterator = iter(json.loads(line) for line in lines)
    else:
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True, cache_dir="/dbfs/")
        ds_iterator = iter(dataset)

    # -------------------------------
    # Get T prompts of fixed length
    # -------------------------------
    prompts = []
    itm = 0
    while itm < T:
        example = next(ds_iterator)
        text = example['text']
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        if len(tokens) < prompt_tokens + buffer_tokens:
            continue
        prompt = tokens[-(buffer_tokens+prompt_tokens):-buffer_tokens]
        prompts.append(prompt)
        itm += 1
    prompts = torch.vstack(prompts)
    print("Successfully loaded dataset...\n")

    # -------------------------------
    # Instantiate the watermark generator
    # -------------------------------
    WG = WatermarkGenerate(model, 
                            vocab_size=vocab_size, 
                            key=args.seed,
                            text_length=args.m, 
                            watermark_type=method, 
                            temperature=temp, 
                            text_window=args.c, 
                            seeding_scheme=args.seed_way)

    # -------------------------------
    # Step 1: Generate watermarked data
    # -------------------------------
    if args.model == "facebook/opt-1.3b":
        model_name = "1p3B"
    else:
        print("No such model name!!!", args.model)
        model_name = "???"

    exp_name = f"text_data/{model_name}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-raw.pkl"
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)

    if os.path.exists(exp_name):
        print(f"Loading data from {exp_name}")
        with open(exp_name, 'rb') as f:
            results = pickle.load(f)
        watermarked_samples = results['watermark']['tokens']
        generated_Ys = results['watermark']['Ys']
        generated_top_probs = results['watermark']['top_probs']
        all_where_watermarks = results['watermark']['where_watermark']
    else:
        print(f"{exp_name} not found. Generating data...")
        results = defaultdict(dict)
        results['args'] = copy.deepcopy(args)
        results['prompts'] = copy.deepcopy(prompts)

        t1 = time()
        watermarked_samples = []
        generated_Ys = []
        generated_top_probs = []
        all_where_watermarks = []

        for batch in tqdm(range(n_batches)):
            idx = torch.arange(batch * batch_size, min(T, (batch + 1) * batch_size))
            generated_tokens, Ys, top_probs, where_watermarks = WG(prompts[idx], 1.)

            watermarked_samples.append(generated_tokens[:, prompt_tokens:])
            generated_Ys.append(Ys)
            generated_top_probs.append(top_probs)
            all_where_watermarks.append(where_watermarks)

        watermarked_samples = torch.cat(watermarked_samples, axis=0)
        generated_Ys = torch.cat(generated_Ys, axis=0)
        generated_top_probs = torch.cat(generated_top_probs, axis=0)
        all_where_watermarks = torch.cat(all_where_watermarks, axis=0)

        results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
        results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
        results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)
        results['watermark']['where_watermark'] = copy.deepcopy(all_where_watermarks)

        print(f'Generated samples in (t = {time()-t1} seconds)')
        pickle.dump(results, open(exp_name, "wb"))

    # -------------------------------
    # Step 2: Apply corruption (substitution, deletion, insertion)
    # -------------------------------
    from attacks import substitution_attack, insertion_attack, deletion_attack

    def corrupt(tokens, text_window, substitution, deletion, insertion):
        if substitution > 0:
            tokens, modify_mask = substitution_attack(tokens, substitution, text_window, eff_vocab_size)
        elif deletion > 0:
            tokens, modify_mask = deletion_attack(tokens, deletion, text_window)
        elif insertion > 0:
            tokens, modify_mask = insertion_attack(tokens, insertion, text_window, eff_vocab_size)
        else:
            modify_mask = torch.zeros(len(tokens), dtype=torch.bool)
        return tokens, modify_mask

    def get_currpted_data(watermarked_samples, text_window, substitution, deletion, insertion, new_tokens):
        watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size - 1)
        corrupted_watermark_data = []
        modify_mask_data = []
        for itm in tqdm(range(args.T), position=0, leave=True):
            sample = watermarked_samples[itm]
            corrupted_sample, modify_mask = corrupt(sample, text_window, substitution, deletion, insertion)
            if len(corrupted_sample) <= new_tokens:
                corrupted_sample = torch.nn.functional.pad(corrupted_sample, (new_tokens - len(corrupted_sample), 0), "constant", 0)
            else:
                corrupted_sample = corrupted_sample[:new_tokens]
            corrupted_watermark_data.append(corrupted_sample)
            modify_mask_data.append(modify_mask)
        corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
        modify_mask_data = torch.vstack(modify_mask_data)
        return corrupted_watermark_data, modify_mask_data

    # Define corruption budgets
    text_window = args.c
    modify_budgets = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]

    # Apply substitution corruption
    for budget in modify_budgets:
        exp_name = f"text_data/{model_name}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-sub{budget}.pkl"
        os.makedirs(os.path.dirname(exp_name), exist_ok=True)
        if not os.path.exists(exp_name):
            results = defaultdict(dict)
            corrupted_samples, modify_mask = get_currpted_data(watermarked_samples, text_window, budget, 0, 0, 500)
            corrupted_Y = WG.compute_Ys(corrupted_samples, prompts)
            results["tokens"] = copy.deepcopy(corrupted_samples)
            results["where_modify"] = copy.deepcopy(modify_mask)
            results["Ys"] = copy.deepcopy(corrupted_Y)
            pickle.dump(results, open(exp_name, "wb"))

    # Apply deletion corruption
    for budget in modify_budgets:
        exp_name = f"text_data/{model_name}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-del{budget}.pkl"
        os.makedirs(os.path.dirname(exp_name), exist_ok=True)
        if not os.path.exists(exp_name):
            results = defaultdict(dict)
            corrupted_samples, modify_mask = get_currpted_data(watermarked_samples, text_window, 0, budget, 0, 300)
            corrupted_Y = WG.compute_Ys(corrupted_samples, prompts)
            results["tokens"] = copy.deepcopy(corrupted_samples)
            results["where_modify"] = copy.deepcopy(modify_mask)
            results["Ys"] = copy.deepcopy(corrupted_Y)
            pickle.dump(results, open(exp_name, "wb"))

    # Apply insertion corruption
    for budget in modify_budgets:
        exp_name = f"text_data/{model_name}-{method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-ins{budget}.pkl"
        os.makedirs(os.path.dirname(exp_name), exist_ok=True)
        if not os.path.exists(exp_name):
            results = defaultdict(dict)
            corrupted_samples, modify_mask = get_currpted_data(watermarked_samples, text_window, 0, 0, budget, 500)
            corrupted_Y = WG.compute_Ys(corrupted_samples, prompts)
            results["tokens"] = copy.deepcopy(corrupted_samples)
            results["where_modify"] = copy.deepcopy(modify_mask)
            results["Ys"] = copy.deepcopy(corrupted_Y)
            pickle.dump(results, open(exp_name, "wb"))

if __name__ == "__main__":
    run(args.method, args.temp)