"""Function validation tests."""

import numpy as np
import pytest
import tensorflow as tf
import torch

from py_benchmark_functions import Function
from py_benchmark_functions.factory import Registry
from py_benchmark_functions.imp import tensorflow as tff, torch as torchf

from .utils import EvaluationSamples, batch_value, to_tensor_or_array


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
        x = to_tensor_or_array(f, x)
        fx = to_tensor_or_array(f, fx)

        # Maybe batch input
        x = batch_value(x, batch_size)
        fx = batch_value(fx, batch_size)

        # Run function
        out = f(x)

        # If torch, might be on GPU
        if isinstance(f, torchf.TorchFunction):
            out = out.cpu()

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


@pytest.mark.parametrize("fn_name", Registry.functions)
@pytest.mark.parametrize("batch_size", [0, 1, 32, 256])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA-capable device."
)
def test_torch_on_gpu(fn_name: str, batch_size: int):
    fn = Registry[fn_name, "torch"](4, device=torch.device("cuda:0"))
    _run_test(fn, batch_size)
