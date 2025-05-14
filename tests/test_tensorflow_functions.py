"""Function validation tests."""

import numpy
import tensorflow as tf

from py_benchmark_functions.imp import tensorflow as tff
from .utils import FunctionEvaluationExamples


batch_size = 10  # batch size of array in multiple input testing
dtype = tf.float32
dims = 4


def _batch_result(array_result: tf.Tensor) -> tf.Tensor:
    return tf.repeat(tf.expand_dims(array_result, 0), batch_size, 0)


def _full_test(f: tff.TensorflowFunction, relax_batch=False, tolerance: float = 10):
    f.disable_tf_function()
    _default_test(f, relax_batch=relax_batch, tolerance=tolerance)

    f.enable_tf_function()
    _default_test(f, relax_batch, tolerance)


def _default_test(f: tff.TensorflowFunction, relax_batch=False, tolerance: float = 10):
    tol = tolerance * numpy.finfo(dtype.as_numpy_dtype).eps

    array, array_result = FunctionEvaluationExamples.get_default_eval(f, dims)
    array = tf.constant(array, dtype=dtype)
    array_result = tf.constant(array_result, dtype)

    # Test default value [1,2,3,4]
    result = f(array)
    tf.debugging.assert_near(result, array_result, tol, tol)

    if not relax_batch:
        batch = tf.repeat(array[None], batch_size, 0)
        bresult = _batch_result(array_result)

        # Test batch of default value [[1,2,3,4],[1,2,3,4], ...]
        result = f(batch)
        tf.debugging.assert_near(result, bresult)
        assert result.shape == bresult.shape

    zero, zero_result = FunctionEvaluationExamples.get_eval_at_zeros(f, dims)
    zero = tf.constant(zero, dtype=dtype)
    zero_result = tf.constant(zero_result, dtype=dtype)

    result = f(zero)
    tf.debugging.assert_near(result, zero_result, tol, tol)

    # Testing shape and dtype
    assert result.shape == array_result.shape
    assert result.dtype == array_result.dtype


def test_ackley():
    _full_test(tff.AckleyTensorflow(dims))


def test_griewank():
    _full_test(tff.GriewankTensorflow(dims))


def test_rastrigin():
    _full_test(tff.RastriginTensorflow(dims))


def test_levy():
    _full_test(tff.LevyTensorflow(dims))


def test_rosenbrock():
    _full_test(tff.RosenbrockTensorflow(dims))


def test_zakharov():
    _full_test(tff.ZakharovTensorflow(dims))


def test_sum_squares():
    _full_test(tff.SumSquaresTensorflow(dims))


def test_sphere():
    _full_test(tff.SphereTensorflow(dims))


def test_streched_v_sine_wave():
    _full_test(tff.StrechedVSineWaveTensorflow(dims), True, 1e5)


def test_bent_cigar():
    _full_test(tff.BentCigarTensorflow(dims), True)


def test_schumer_steiglitz():
    _full_test(tff.SchumerSteiglitzTensorflow(dims), True)


def test_powell_sum():
    _full_test(tff.PowellSumTensorflow(dims), True)


def test_alpine_2():
    _full_test(tff.Alpine2Tensorflow(dims), True)


def test_csendes():
    _full_test(tff.CsendesTensorflow(dims), True)


def test_deb_1():
    _full_test(tff.Deb1Tensorflow(dims), True)


def test_deb_3():
    _full_test(tff.Deb3Tensorflow(dims), True)


def test_qing():
    _full_test(tff.QingTensorflow(dims), True)


def test_sargan():
    _full_test(tff.SarganTensorflow(dims), True)


def test_whitley():
    _full_test(tff.WhitleyTensorflow(dims), True)


def test_schwefel():
    _full_test(tff.SchwefelTensorflow(dims), True)


def test_chung_reynolds():
    _full_test(tff.ChungReynoldsTensorflow(dims), True)


def test_schwefel_2_26():
    _full_test(tff.Schwefel226Tensorflow(dims), True)


def test_schwefel_1_2():
    _full_test(tff.Schwefel12Tensorflow(dims), True)


def test_schwefel_2_22():
    _full_test(tff.Schwefel222Tensorflow(dims), True)


def test_schwefel_2_23():
    _full_test(tff.Schwefel223Tensorflow(dims), True)


def test_brown():
    _full_test(tff.BrownTensorflow(dims), True)


def test_salomon():
    _full_test(tff.SalomonTensorflow(dims), True)


def test_trigonometric_2():
    _full_test(tff.Trigonometric2Tensorflow(dims), True)


def test_mishra_2():
    _full_test(tff.Mishra2Tensorflow(dims), True)


def test_weierstrass():
    _full_test(tff.WeierstrassTensorflow(dims), True, 1e3)


def test_w_wavy():
    _full_test(tff.WWavyTensorflow(dims), True)


def test_rotated_hyper_ellipsoid():
    _full_test(tff.RotatedHyperEllipsoidTensorflow(dims))


def test_dixon_price():
    _full_test(tff.DixonPriceTensorflow(dims))
