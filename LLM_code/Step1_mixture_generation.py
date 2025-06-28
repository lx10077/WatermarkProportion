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

# ======================= Argument Parsing ============================
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="Gumbel", type=str)  # Watermark method to use
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)  # HuggingFace model
parser.add_argument('--seed', default=15485863, type=int)  # Random seed for reproducibility
parser.add_argument('--c', default=4, type=int)  # Window size for watermark seeding
parser.add_argument('--m', default=500, type=int)  # Number of new tokens to generate
parser.add_argument('--T', default=500, type=int)  # Number of total prompts
parser.add_argument('--temp', default=0.7, type=float)  # Sampling temperature

parser.add_argument('--seed_way', default="noncomm_prf", type=str)  # PRF seeding scheme

parser.add_argument('--prompt_tokens', default=50, type=int)  # Length of prompt before generation
parser.add_argument('--buffer_tokens', default=20, type=int)  # Tokens held out from prompt for context
parser.add_argument('--max_seed', default=100000, type=int)  # (Unused)
parser.add_argument('--norm', default=1, type=int)  # (Unused)
parser.add_argument('--rt_translate', action='store_true')  # (Unused)
parser.add_argument('--language', default="french", type=str)  # (Unused)
parser.add_argument('--truncate_vocab', default=8, type=int)  # How many tokens to exclude from vocab

args = parser.parse_args()
print(args)

# ==================== Initialization ============================
t0 = time()
torch.manual_seed(args.seed)  # Set random seed

print(f"Using {torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", offload_folder="./offload_folder")
model.eval()

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab  # Adjusted vocabulary size

print(f'Loaded the model (t = {time()-t0} seconds)')

# =================== Prompt Preparation ===========================
T = args.T
batch_size = 10 if args.method == "Gumbel" else 1
n_batches = int(np.ceil(T / batch_size))
new_tokens = args.m
load_local_data = True
buffer_tokens = args.buffer_tokens
prompt_tokens = args.prompt_tokens

if load_local_data:
    # Load from local file instead of dataset library
    with open('c4/c4.json', 'r') as f:
        lines = f.readlines()
    ds_iterator = iter(json.loads(line) for line in lines)
else:
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True, cache_dir="/dbfs/")
    ds_iterator = iter(dataset)

# Collect T prompt sequences from dataset
prompts = []
itm = 0
while itm < T:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
    if len(tokens) < prompt_tokens + buffer_tokens:
        continue
    prompt = tokens[-(buffer_tokens+prompt_tokens):-buffer_tokens]  # Extract desired segment
    prompts.append(prompt)

    itm += 1

prompts = torch.vstack(prompts)
print("Successully loading dataset...\n")

# ==================== Initialize Watermark Generator ====================
WG = WatermarkGenerate(
    model=model,
    vocab_size=vocab_size,
    key=args.seed,
    text_length=args.m,
    watermark_type=args.method,
    temperature=args.temp,
    text_window=args.c,
    seeding_scheme=args.seed_way
)

# =================== Generate and Save Data ============================
def generate_mixture_data(eps):
    print(eps)
    results = defaultdict(dict)
    results['args'] = copy.deepcopy(args)
    results['prompts'] = copy.deepcopy(prompts)

    t1 = time()
    watermarked_samples = []
    generated_Ys = []
    generated_top_probs = []
    all_where_watermarks = []

    # Generate in batches
    for batch in tqdm(range(n_batches)):
        idx = torch.arange(batch * batch_size, min(T, (batch + 1) * batch_size))

        # Generate tokens and collect watermark metadata
        generated_tokens, Ys, top_probs, where_watermarks = WG(prompts[idx], eps)

        watermarked_samples.append(generated_tokens[:, prompt_tokens:])
        generated_Ys.append(Ys)
        generated_top_probs.append(top_probs)
        all_where_watermarks.append(where_watermarks)

    # Combine batch results
    watermarked_samples = torch.cat(watermarked_samples, axis=0)
    generated_Ys = torch.cat(generated_Ys, axis=0)
    generated_top_probs = torch.cat(generated_top_probs, axis=0)
    all_where_watermarks = torch.cat(all_where_watermarks, axis=0)

    results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
    results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
    results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)
    results['watermark']['where_watermark'] = copy.deepcopy(all_where_watermarks)

    print(f'Generated samples in (t = {time()-t1} seconds)')

    # Determine output name based on model
    if args.model == "facebook/opt-1.3b":
        model_name = "1p3B"
    else:
        model_name = args.model.split("/")[-1]

    # Save results to file
    exp_name = f"text_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{args.temp}-eps{eps}.pkl"
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)
    pickle.dump(results, open(exp_name, "wb"))


if __name__ == "__main__":
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        generate_mixture_data(eps)
