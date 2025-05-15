"""Function validation tests."""

from typing import List

import numpy as np
import pytest
import tensorflow as tf

from py_benchmark_functions import Function
from py_benchmark_functions.factory import Registry
from py_benchmark_functions.imp import tensorflow as tff

from .utils import EvaluationSamples


def _batch_value(value, batch_size: int):
    if batch_size <= 0:
        return value

    if tf.is_tensor(value):
        return tf.repeat(tf.expand_dims(value, 0), batch_size, 0)

    return np.repeat(np.expand_dims(value, 0), batch_size, 0)


def _to_tensor_or_array(fn: Function, value: List[float]):
    if isinstance(fn, tff.TensorflowFunction):
        return tf.constant(value, dtype=tf.float32)

    return np.array(value, dtype=np.float32)


def _run_test(f: Function, batch_size: int, tol: float = 0.005):
    dims = f.dims
    global_optimum = (
        []
        if f.metadata.global_optimum is None
        or f.metadata.global_optimum_coordinates is None
        else [
            f.metadata.concrete_optimum_coordinates(dims),
            f.metadata.concrete_optimum(dims),
        ]
    )

    for v, should_batch in zip(
        [
            EvaluationSamples.sample_eval(f, dims),
            EvaluationSamples.sample_at_zeroes(f, dims),
            global_optimum,
        ],
        [True, False],
    ):
        if not v:
            continue

        # Read values
        x, fx = v

        # Convert to Tensors
        x = _to_tensor_or_array(f, x)
        fx = _to_tensor_or_array(f, fx)

        # Maybe batch input
        x = _batch_value(x, batch_size)
        fx = _batch_value(fx, batch_size)

        # Run function
        out = f(x)

        # Shape and dtype assertion
        assert out.shape == fx.shape
        assert out.dtype == fx.dtype

        # Assert values
        if isinstance(f, tff.TensorflowFunction):
            tf.debugging.assert_near(
                out,
                fx,
                tol,
                tol,
            )
        else:
            assert np.isclose(out, fx, atol=tol).all()


@pytest.mark.parametrize("fn_name", Registry.functions)
@pytest.mark.parametrize("backend", Registry.backends)
@pytest.mark.parametrize("batch_size", [0, 1, 32, 256])
def test_functions(fn_name: str, backend: str, batch_size: int):
    # Currently only 4 dimensions are supported
    fn = Registry[fn_name, backend](4)

    # Run tests in those conditions
    _run_test(fn, batch_size)
