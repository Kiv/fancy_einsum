import numpy as np
from numpy import allclose
from hypothesis import given
from hypothesis.strategies import integers, composite, lists
from hypothesis.extra.numpy import arrays

from fancy_einsum import einsum

def tensor(draw, shape):
    return draw(arrays(dtype=int, shape=shape))

@composite
def square_matrix(draw):
    n = draw(integers(2, 10))
    return tensor(draw, (n, n))


@given(square_matrix())
def test_simple_matmul(mat):
    actual = einsum('length length ->', mat)
    assert allclose(actual, np.einsum('aa->', mat))


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
    assert allclose(actual, np.einsum('...rt,...tc->...rc', a, b))