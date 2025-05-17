"""Function validation tests."""

import numpy as np
import pytest

from py_benchmark_functions.factory import Registry

from .utils import batch_value, to_tensor_or_array


@pytest.mark.parametrize("fn_name", Registry.functions)
@pytest.mark.parametrize("backend", ["torch", "tensorflow"])
@pytest.mark.parametrize("batch_size", [0, 1, 256])
def test_grads_dtype_shape(fn_name: str, backend: str, batch_size: int):
    # Select function for two dimensions
    fn = Registry[fn_name, backend](2)

    # Allocate vector
    x = to_tensor_or_array(fn, [1.0, 2.0])

    # Maybe apply batch_size
    x = batch_value(x, batch_size)

    # Evaluate function and get grads
    grads, _ = fn.grads_at(x)

    # Both tensorflow and torch allow for
    #   numpy conversion
    grads = grads.numpy()

    # Check that both methods produce the same
    #   gradients
    assert (fn.grads(x).numpy() == grads).all()

    # Assert shape matches expected
    assert grads.shape == x.numpy().shape

    # Assert float dtype
    assert np.issubdtype(grads.dtype, np.floating)
