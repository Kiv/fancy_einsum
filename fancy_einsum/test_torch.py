import itertools
import string

import pytest
import torch
from torch import allclose
from hypothesis import given
from hypothesis.strategies import integers, composite, lists
from hypothesis.extra.numpy import arrays

from fancy_einsum import einsum

def tensor(draw, shape):
    arr = draw(arrays(dtype=int, elements=integers(-10,10), shape=shape))
    return torch.Tensor(arr)

@composite
def square_matrix(draw):
    n = draw(integers(2, 10))
    return tensor(draw, (n, n))


@given(square_matrix())
def test_simple_matmul(mat):
    actual = einsum('length length ->', mat)
    assert allclose(actual, torch.einsum('aa->', mat))


@composite
def matmul_compatible(draw):
    b = draw(integers(1, 10))
    r = draw(integers(1, 10))
    t = draw(integers(1, 10))
    c = draw(integers(1, 10))
    return tensor(draw, (b, r, t)), tensor(draw, (b, t, c))


@given(matmul_compatible())
def test_ellipse_matmul(args):
    a, b = args
    actual = einsum('...rows temp, ...temp cols -> ...rows cols', a, b)
    assert allclose(actual, torch.einsum('...rt,...tc->...rc', a, b))


@composite
def chain_matmul(draw):
    sizes = [draw(integers(1, 4)) for _ in range(5)]
    shapes = [(sizes[i-1], sizes[i]) for i in range(1, len(sizes))]
    return [tensor(draw, shape) for shape in shapes]


@given(chain_matmul())
def test_chain_matmul(args):
    actual = einsum('rows t1, t1 t2, t2 t3, t3 cols -> rows cols', *args)
    assert allclose(actual, torch.einsum('ab,bc,cd,de->ae', *args))


def test_too_many_variables():
    names = [''.join(p) for p in itertools.permutations(string.ascii_lowercase, 3)]
    eq = f'{" ".join(names)} ->'
    print(eq)
    tensor = torch.ones([1] * len(names))
    with pytest.raises(ValueError):
        einsum(eq, tensor)


    


if __name__ == '__main__':
    import pytest
    pytest.main()
