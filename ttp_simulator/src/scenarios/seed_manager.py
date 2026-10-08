import random

def set_global_seed(seed: int) -> None:
    random.seed(seed)

def get_random_seed() -> int:
    return random.randint(0, 2**32 - 1)
