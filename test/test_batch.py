import os
import sys

import pytest
from argparse import Namespace
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from batch import Batch

""" ====================================== INSTANTIATION ====================================== """


def test_batch_init():
    test_batch = Batch(a=1, b=2)

    assert test_batch.__dict__["a"] == 1
    assert test_batch.__dict__["b"] == 2


def test_batch_dict():
    data = {"a": 1, "b": 2}
    test_batch = Batch.from_dict(data)

    assert test_batch.__dict__["a"] == 1
    assert test_batch.__dict__["b"] == 2


""" ====================================== INDEXING ====================================== """


def test_batch_get_str():
    test_batch = Batch(a=1, b=2)

    assert test_batch["a"] == 1
    assert test_batch["b"] == 2


def test_batch_get_str_tuple():
    test_batch = Batch(a=1, b=2)
    batch_out = test_batch["a", "b"]

    assert batch_out["a"] == 1
    assert batch_out["b"] == 2


def test_batch_get_int():
    test_batch = Batch(a=[1, 2], b=[3, 4])
    batch_out = test_batch[1]

    assert batch_out["a"] == 2
    assert batch_out["b"] == 4


def test_batch_get_int_slice():
    test_batch = Batch(a=np.arange(9).reshape(3,3), b=np.arange(10,19).reshape(3,3))
    batch_out = test_batch[:, 2]

    assert np.allclose(batch_out["a"], np.array([2, 5, 8]))
    assert np.allclose(batch_out["b"], np.array([12, 15, 18]))


""" ====================================== OPERATORS ====================================== """


@pytest.mark.parametrize("op", [
    # "__not__",
    "__abs__", "__index__",
    # "__inv__",
    "__invert__", "__neg__", "__pos__",
])
def test_batch_operation_unary(op):
    a_1, b_1 = 1, 2

    test_batch_1 = Batch(a=a_1, b=b_1)

    test_batch_3 = getattr(test_batch_1, op)()
    assert test_batch_3.a == getattr(a_1, op)()
    assert test_batch_3.b == getattr(b_1, op)()


@pytest.mark.parametrize("op", [
    "__add__", "__and__",
    # "__concat__",
    "__floordiv__", "__lshift__", "__mod__", "__mul__",
    "__or__", "__pow__", "__rshift__", "__sub__", "__truediv__", "__xor__",
    # "__contains__",
])
def test_batch_operation_binary(op):
    a_1, b_1 = 1, 2
    a_2, b_2 = 3, 4

    test_batch_1 = Batch(a=a_1, b=b_1)
    test_batch_2 = Batch(a=a_2, b=b_2)

    test_batch_3 = getattr(test_batch_1, op)(test_batch_2)
    assert test_batch_3.a == getattr(a_1, op)(a_2)
    assert test_batch_3.b == getattr(b_1, op)(b_2)


def test_batch_member_value():
    test_batch_1 = Batch(a=Namespace(x=1), b=Namespace(x=2))

    test_batch_2 = test_batch_1.x
    assert test_batch_2.a == 1
    assert test_batch_2.b == 2
