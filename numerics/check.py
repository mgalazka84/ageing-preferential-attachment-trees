"""Check the rejection probabilities on every recursive tree through size eight."""
from itertools import product
from math import isclose


def check_tree(n, parents, alpha):
    age_weight = [0.] + [j ** (-alpha) for j in range(1, n+1)]
    denom = 2 * sum(age_weight)
    degree = [0] + [1] * n
    proposal_mass = [0.] * (n+1)
    for parent in parents:
        degree[parent] += 1
    for age in range(1, n+1):
        slot = n-age+1
        proposal_mass[slot] += age_weight[age]/denom
        if 2 <= slot < n:
            parent = parents[slot-2]
            success = age_weight[n-parent+1]/age_weight[age]
            assert 0 < success <= 1
            proposal_mass[parent] += age_weight[age] * success/denom
    direct = [degree[v] * age_weight[n-v+1]/denom for v in range(1, n+1)]
    assert all(isclose(p,q,rel_tol=5.e-14) for p,q in zip(proposal_mass[1:],direct))
    assert .5-1.e-15 <= sum(direct) <= 1+1.e-15


if __name__ == '__main__':
    total = 0
    for n in range(2, 9):
        for history in product(*(range(1, m+1) for m in range(2, n))):
            for alpha in (0., .25, .5, .75, .9):
                check_tree(n, history, alpha)
            total += 1
    print(f'Verified one-step probabilities on {total} trees at five ageing exponents.')
