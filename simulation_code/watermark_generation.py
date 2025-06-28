import numpy as np
import random

### Gumbel-max watermark utilities

def Zipf(a=1., b=0.01, support_size=5):
    support_Ps = np.arange(1, 1 + support_size)
    support_Ps = (support_Ps + b) ** (-a)
    support_Ps /= support_Ps.sum()
    return support_Ps

def random_Ps(K):
    a = np.random.uniform(0.95, 1.5)
    b = np.random.uniform(0.01, 0.5)
    return Zipf(a=a, b=b, support_size=K)

def rng(key, m=2**31, a=1103515245, c=12345):
    return ((a * key + c) % m) / (m - 1)

def dominate_Ps(Delta, K):
    a = np.random.uniform(0.95, 1.5)
    b = np.random.uniform(0.01, 0.1)
    support_size = np.random.randint(low=5, high=15)

    Head_Ps = Zipf(a=a, b=b, support_size=support_size)
    b = (1 - Delta) / Head_Ps.max()
    Ps = np.ones(K)

    if b <= 1:
        Tail_Ps = np.ones(K - support_size) / (K - support_size)
        Ps[:support_size] = b * Head_Ps
        Ps[support_size:] = (1 - b) * Tail_Ps
    else:
        Ps[0] = 1 - Delta
        Ps[1:1 + support_size] = Head_Ps * Delta / 2
        Tail_Ps = np.ones(K - support_size - 1) / (K - support_size - 1)
        Ps[1 + support_size:] = Delta / 2 * Tail_Ps

    assert Ps.max() <= 1 - Delta + 1e-5 and np.abs(np.sum(Ps) - 1) <= 1e-3
    return Ps

def generate_gumbel_uniform_local(inputs, c, key, K):
    assert len(inputs) >= c - 1
    seed_input = tuple(inputs[-(c - 1):] + [key])
    seed = hash(seed_input) % (2**32)
    np.random.seed(seed)
    return np.random.uniform(size=K)

def generate_gumbel_watermark_text(prompt, K=1000, T=60, c=5, Delta=0.5, key=1):
    inputs = prompt.copy()
    selected_Ys = []
    highest_probs = []

    for _ in range(T):
        if Delta is None:
            Probs = random_Ps(K)
        else:
            if isinstance(Delta, float) and 0 < Delta < 1:
                Probs = dominate_Ps(Delta, K)
            elif isinstance(Delta, str):
                if Delta[-1] == "+":
                    Delta = np.random.uniform(float(Delta[:-1]), 1 - 1e-3)
                else:
                    Delta = np.random.uniform(1e-3, float(Delta[:-1]))
                Probs = dominate_Ps(Delta, K)
            else:
                raise ValueError(f"No support for this Delta: {Delta}!")

        Probs = np.random.permutation(Probs)
        uniform_xi = generate_gumbel_uniform_local(inputs, c, key, K)
        next_token = np.argmax(uniform_xi ** (1 / Probs))

        inputs.append(next_token)
        selected_Ys.append(uniform_xi[next_token])
        highest_probs.append(np.max(Probs))

    return inputs, selected_Ys, highest_probs

def recover_gumbel_xi(prompt, inputs, c):
    initial_length = len(prompt)
    selected_xis = []
    for i in range(initial_length + 1, len(inputs) + 1):
        random.seed(tuple(inputs[(i - c):i]))
        selected_xis.append(random.random())
    return selected_xis

def F(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += Probs[k] * x ** (1 / Probs[k])
    return rho

def f(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += x ** (1 / Probs[k] - 1)
    return rho

### Inverse-transform watermark utilities

def generate_inverse_uniform_local(inputs, c, key, K):
    assert len(inputs) >= c - 1
    seed_input = tuple(inputs[-(c - 1):] + [key])
    seed = hash(seed_input) % (2**32)
    np.random.seed(seed)
    xi = np.random.uniform(size=1)[0]
    pi = np.random.permutation(K)
    return xi, pi

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def find_next_token(xi, probs, pi):
    inv_pi = inv(pi)
    inv_probs = probs[inv_pi]
    i = 0
    s = 0
    while s <= xi:
        s += inv_probs[i]
        i += 1
    return inv_pi[i - 1]

def generate_inverse_watermark_text(prompt, K=1000, T=60, c=5, Delta=0.5, key=1):
    inputs = prompt.copy()
    selected_covs = []
    selected_Us = []
    selected_Pis = []
    selected_difs = []
    highest_probs = []

    for _ in range(T):
        if Delta is None:
            Probs = random_Ps(K)
        else:
            if isinstance(Delta, float) and 0 < Delta < 1:
                Probs = dominate_Ps(Delta, K)
            elif isinstance(Delta, str):
                if Delta[-1] == "+":
                    Delta = np.random.uniform(float(Delta[:-1]), 1 - 1e-3)
                else:
                    Delta = np.random.uniform(1e-3, float(Delta[:-1]))
                Probs = dominate_Ps(Delta, K)
            else:
                raise ValueError(f"No support for this Delta: {Delta}!")

        Probs = np.random.permutation(Probs)
        xi, pi = generate_inverse_uniform_local(inputs, c, key, K)
        next_token = find_next_token(xi, Probs, pi)

        eta = (pi[next_token] - 1) / (K - 1)
        selected_covs.append((xi - 0.5) * (eta - 0.5))
        selected_difs.append(-np.abs(xi - eta))
        selected_Us.append(xi)
        selected_Pis.append(pi[next_token])

        inputs.append(next_token)

    return (
        inputs,
        np.array(selected_covs),
        np.array(selected_difs),
        np.array(selected_Us),
        np.array(selected_Pis),
        highest_probs
    )

def recover_inverse_xi(prompt, inputs, c):
    initial_length = len(prompt)
    selected_xis = []
    for i in range(initial_length + 1, len(inputs) + 1):
        random.seed(tuple(inputs[(i - c):i]))
        selected_xis.append(random.random())
    return selected_xis

### Green-red list watermark utilities

def generate_greenred_uniform_local(inputs, c, key, K):
    assert len(inputs) >= c - 1
    seed_input = tuple(inputs[-(c - 1):] + [key])
    seed = hash(seed_input) % (2**32)
    np.random.seed(seed)
    return np.random.permutation(K)

def modify_NTP(original_NTP, pi, gamma, delta):
    K = len(original_NTP)
    green_size = int(gamma * K)
    green_indices = pi[:green_size]

    modified_NTP = np.copy(original_NTP)
    modified_NTP[green_indices] *= np.exp(delta)
    modified_NTP /= np.sum(modified_NTP)
    return modified_NTP

def sample_from(NTP, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(len(NTP), p=NTP)

def is_green(token, pi, gamma):
    green_size = int(gamma * len(pi))
    greenlist = set(pi[:green_size])
    return int(token in greenlist)

def generate_greenred_watermark_text(prompt, K=1000, T=60, c=5, Delta=0.5, key=1, gamma=0.5, delta=1):
    inputs = prompt.copy()
    highest_probs = []
    selected_Ys = []

    for _ in range(T):
        if Delta is None:
            Probs = random_Ps(K)
        else:
            if isinstance(Delta, float) and 0 < Delta < 1:
                Probs = dominate_Ps(Delta, K)
            elif isinstance(Delta, str):
                if Delta[-1] == "+":
                    Delta = np.random.uniform(float(Delta[:-1]), 1 - 1e-3)
                else:
                    Delta = np.random.uniform(1e-3, float(Delta[:-1]))
                Probs = dominate_Ps(Delta, K)
            else:
                raise ValueError(f"No support for this Delta: {Delta}!")

        pi = generate_greenred_uniform_local(inputs, c, key, K)
        modified_probs = modify_NTP(Probs, pi, gamma, delta)
        next_token = sample_from(modified_probs)
        Y = is_green(next_token, pi, gamma)

        inputs.append(next_token)
        selected_Ys.append(Y)
        highest_probs.append(np.max(Probs))

    return inputs, selected_Ys, highest_probs
