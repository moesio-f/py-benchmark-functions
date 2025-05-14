"""Function validation tests."""

import numpy as np

from py_benchmark_functions.imp import numpy as npf
from .utils import FunctionEvaluationExamples


batch_size = 2  # batch size of array in multiple input testing
dtype = np.float64
dims = 4


def _default_test(f: npf.NumpyFunction):
    array, array_result = FunctionEvaluationExamples.get_default_eval(f, dims)
    array = np.array(array, dtype)
    array_result = np.array(array_result, dtype)

    # Test default value [1,2,3,4]
    result = f(array)
    assert result == array_result

    batch = array[None].repeat(batch_size, axis=0)
    batch_result = np.array(array_result).repeat(batch_size)

    # Test batch of default value [[1,2,3,4],[1,2,3,4], ...]
    result = f(batch)
    assert np.array_equal(result, batch_result)
    assert result.shape == batch_result.shape

    zero, zero_result = FunctionEvaluationExamples.get_eval_at_zeros(f, dims)
    zero = np.array(zero, dtype=dtype)
    zero_result = np.array(zero_result, dtype=dtype)

    result = f(zero)
    assert result == zero_result

    # Testing shape and dtype
    assert result.shape == zero_result.shape
    assert result.dtype == zero_result.dtype


def test_ackley():
    _default_test(npf.AckleyNumpy(dims))


def test_griewank():
    _default_test(npf.GriewankNumpy(dims))


def test_rastrigin():
    _default_test(npf.RastriginNumpy(dims))


def test_levy():
    _default_test(npf.LevyNumpy(dims))


def test_rosenbrock():
    _default_test(npf.RosenbrockNumpy(dims))


def test_zakharov():
    _default_test(npf.ZakharovNumpy(dims))


def test_sum_squares():
    _default_test(npf.SumSquaresNumpy(dims))


def test_sphere():
    _default_test(npf.SphereNumpy(dims))


def test_rotated_hyper_ellipsoid():
    _default_test(npf.RotatedHyperEllipsoidNumpy(dims))


def test_dixon_price():
    _default_test(npf.DixonPriceNumpy(dims))
