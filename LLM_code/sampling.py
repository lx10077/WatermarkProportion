import torch
from IPython import embed
from alternative_prf_schemes import prf_lookup

def seed_rng(generator, tokens, seeding_scheme="minhash_prf", hash_key=15485863, c=5):
    """
    Seed the random number generator using a context-based PRF hash.

    Args:
        generator: PyTorch random generator.
        tokens (Tensor): Input tokens with shape (1, current_length).
        seeding_scheme (str): PRF type, e.g., "minhash_prf".
        hash_key (int): Salt for the PRF.
        c (int): Number of context tokens used for hashing.
    """
    assert tokens.shape[-1] >= c, f"seeding_scheme={seeding_scheme} requires at least {c} context tokens"
    prf_key = prf_lookup[seeding_scheme](tokens[0][-c:], salt_key=hash_key)
    generator.manual_seed(prf_key)

######################################
# Gumbel-Max Watermarking
######################################

def gumbel_key_func(generator, inputs, vocab_size, key, c, seeding_scheme):
    """
    Generate Gumbel noise xi and identity permutation for Gumbel-Max watermarking.

    Returns:
        xis: Tensor of shape (batch_size, vocab_size)
        pis: Tensor of shape (batch_size, vocab_size)
    """
    xis = []
    pis = []
    for k in range(inputs.shape[0]):
        seed_rng(generator, inputs[k].unsqueeze(0), seeding_scheme=seeding_scheme, hash_key=key, c=c)
        xi = torch.rand(size=(1, vocab_size), generator=generator)
        pi = torch.arange(vocab_size)
        xis.append(xi)
        pis.append(pi)
    return torch.vstack(xis), torch.vstack(pis)

def gumbel_sampling(probs, pi, xi):
    """
    Sample next token index via Gumbel-Max trick.
    """
    return torch.argmax(xi ** (1 / torch.gather(probs, 1, pi)), axis=1).unsqueeze(-1)

def gumbel_Y(s, pi, xi):
    """
    Retrieve the Gumbel noise value corresponding to the sampled token.
    """
    return torch.gather(xi, -1, s.cpu()).squeeze()

######################################
# Inverse-Transform Watermarking
######################################

def transform_key_func(generator, inputs, vocab_size, key, c, seeding_scheme):
    """
    Generate xi and pi for inverse-transform watermarking.

    Returns:
        xis: Tensor of shape (batch_size, 1)
        pis: Tensor of shape (batch_size, vocab_size)
    """
    batch_size = inputs.shape[0]
    xis, pis = [], []
    for _ in range(batch_size):
        seed_rng(generator, inputs, seeding_scheme=seeding_scheme, hash_key=key, c=c)
        xi = torch.rand(size=(batch_size, 1), generator=generator)
        pi = torch.randperm(vocab_size, generator=generator)
        xis.append(xi)
        pis.append(pi)
    return torch.vstack(xis), torch.vstack(pis)

def inverse_permutation(perm):
    """
    Compute inverse permutation for a 1D tensor.
    """
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

def transform_sampling(probs, pi, xi):
    """
    Perform inverse transform sampling using CDF and sorted permutation.
    """
    inv_pi = inverse_permutation(pi.squeeze()).unsqueeze(0)
    cdf = torch.cumsum(torch.gather(probs, 1, inv_pi), 1)
    return torch.gather(inv_pi, 1, torch.searchsorted(cdf, xi))

def transform_Y_dif(s, pi, xi):
    """
    Compute distance-based score between xi and quantile of selected token.
    """
    vocab_size = pi.shape[1]
    s_samp = torch.gather(pi, -1, s.cpu()).squeeze()
    return -torch.abs(xi - (s_samp - 1) / (vocab_size - 1))
