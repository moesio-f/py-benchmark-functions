"""Transformations validation tests."""

import numpy as np
import tensorflow as tf
import torch

from py_benchmark_functions.imp import numpy as npf
from py_benchmark_functions.imp import tensorflow as tff
from py_benchmark_functions.imp import torch as torchf

batch_size = 10  # batch size of array in multiple input testing


class DummyNumpyFunction(npf.NumpyFunction):
    def __init__(self, dims: int):
        super().__init__(dims, -100.0, 100.0)

    def _fn(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x)


class DummyTensorflowFunction(tff.TensorflowFunction):
    def __init__(self, dims: int):
        super().__init__(dims, -100.0, 100.0)

    def _fn(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(x)


class DummyTorchFunction(torchf.TorchFunction):
    def __init__(self, dims: int):
        super().__init__(dims, -100.0, 100.0)

    def _fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)


def test_np_composition():
    fn = DummyNumpyFunction(3)
    tfn = npf.NumpyTransformation(fn, inner_scale=2.0, outer_scale=1.0)
    tfn = npf.NumpyTransformation(tfn, vshift=1.0)
    tfn = npf.NumpyTransformation(tfn, hshift=1.0)
    tfn = npf.NumpyTransformation(tfn, inner_scale=1.0, outer_scale=0.5)
    arr = np.array([1.0, 1.0, 1.0])

    assert fn(arr) == 3.0
    assert tfn(arr) == 6.5


def test_tf_composition():
    fn = DummyTensorflowFunction(3)
    tfn = tff.TensorflowTransformation(fn, inner_scale=2.0, outer_scale=1.0)
    tfn = tff.TensorflowTransformation(tfn, vshift=1.0)
    tfn = tff.TensorflowTransformation(tfn, hshift=1.0)
    tfn = tff.TensorflowTransformation(tfn, inner_scale=1.0, outer_scale=0.5)
    arr = tf.constant([1.0, 1.0, 1.0])

    fn.disable_tf_function()
    tfn.disable_tf_function()
    assert fn(arr) == 3.0
    assert tfn(arr) == 6.5

    fn.enable_tf_function()
    tfn.enable_tf_function()
    assert fn(arr) == 3.0
    assert tfn(arr) == 6.5


def test_torch_composition():
    fn = DummyTorchFunction(3)
    tfn = torchf.TorchTransformation(fn, inner_scale=2.0, outer_scale=1.0)
    tfn = torchf.TorchTransformation(tfn, vshift=1.0)
    tfn = torchf.TorchTransformation(tfn, hshift=1.0)
    tfn = torchf.TorchTransformation(tfn, inner_scale=1.0, outer_scale=0.5)
    arr = torch.tensor([1.0, 1.0, 1.0])
    for dev in ["cpu"] + ["cuda:0"] if torch.cuda.is_available() else []:
        fn.set_device(torch.device(dev))
        assert fn(arr).cpu() == 3.0
        assert tfn(arr).cpu() == 6.5
