import torch

def substitution_attack(tokens, p, m, vocab_size, distribution=None):
    """
    Randomly substitute `p` tokens in `tokens` with samples from a given distribution.

    Args:
        tokens (torch.Tensor): Input token sequence.
        p (int): Number of tokens to substitute.
        m (int): Length of the window after substitution to mark as corrupted.
        vocab_size (int): Size of the vocabulary.
        distribution (callable, optional): A function that returns a (len(tokens), vocab_size)
            tensor of probabilities for each token.

    Returns:
        (torch.Tensor, torch.BoolTensor): Modified tokens, corruption mask.
    """
    if p == 0:
        return tokens, torch.zeros(len(tokens), dtype=torch.bool)

    tokens = tokens.clone()
    if distribution is None:
        distribution = lambda x: torch.ones((len(x), vocab_size)) / vocab_size

    idx = torch.randperm(len(tokens))[:p]
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1).flatten()
    tokens[idx] = samples[idx]

    modified_mask = torch.zeros(len(tokens), dtype=torch.bool)
    for i in idx:
        end = min(len(tokens), i + m + 1)
        modified_mask[i:end] = True

    return tokens, modified_mask


def deletion_attack(tokens, p, m):
    """
    Randomly delete `p` tokens from `tokens` and mark a window of length `m` after each deletion as corrupted.

    Args:
        tokens (torch.Tensor): Input token sequence.
        p (int): Number of tokens to delete.
        m (int): Length of the window after deletion to mark as corrupted.

    Returns:
        (torch.Tensor, torch.BoolTensor): Modified tokens, corruption mask.
    """
    if p == 0:
        return tokens, torch.zeros(len(tokens), dtype=torch.bool)

    idx = torch.randperm(len(tokens))[:p]
    keep = torch.ones(len(tokens), dtype=torch.bool)
    keep[idx] = False

    corruption_mask = torch.zeros(len(tokens), dtype=torch.bool)
    for i in idx:
        end = min(len(tokens), i + m + 1)
        corruption_mask[i:end] = True

    tokens = tokens[keep]
    corruption_mask = corruption_mask[keep]

    return tokens, corruption_mask


def insertion_attack(tokens, p, m, vocab_size, distribution=None):
    """
    Randomly insert `p` tokens sampled from a given distribution into `tokens`,
    and mark a window of length `m` starting at each insertion point as corrupted.

    Args:
        tokens (torch.Tensor): Input token sequence.
        p (int): Number of tokens to insert.
        m (int): Length of the window after insertion to mark as corrupted.
        vocab_size (int): Size of the vocabulary.
        distribution (callable, optional): A function that returns a (len(tokens), vocab_size)
            tensor of probabilities for each token.

    Returns:
        (torch.Tensor, torch.BoolTensor): Modified tokens, corruption mask.
    """
    if p == 0:
        return tokens, torch.zeros(len(tokens), dtype=torch.bool)

    if distribution is None:
        distribution = lambda x: torch.ones((len(x), vocab_size)) / vocab_size

    idx = torch.randperm(len(tokens))[:p]
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1)

    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i], samples[i], tokens[i:]])

    modified_mask = torch.zeros(len(tokens), dtype=torch.bool)
    for i in idx:
        end = min(len(tokens), i + m + 1)
        modified_mask[i:end] = True

    return tokens, modified_mask
