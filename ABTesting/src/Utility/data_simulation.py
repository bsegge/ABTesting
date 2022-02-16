from numpy.random import randn, rand, binomial


def simulate_data(
        n: int,
        spread_mod: int = 5,
        shift_mod: int = 0,
        distribution: str = "normal",
        prob: float = None
):
    distribution = distribution.lower()
    data = None
    if distribution == "normal":
        data = spread_mod * randn(n) + shift_mod
    elif distribution == "random":
        data = spread_mod * rand(n) + shift_mod
    elif distribution == "binomial":
        successes = binomial(n, prob)
        data = [(successes, 'success'), (n - successes, 'failure')]

    return data
