import numpy as np
import random
import torch
import gc


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()