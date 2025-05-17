from functools import cached_property
from math import e, pi
from typing import List, Tuple, Union

import tensorflow as tf

from py_benchmark_functions import config, core
from py_benchmark_functions.info import FunctionMetadata

# Ensure tf.function's are run as graphs
tf.config.run_functions_eagerly(False)


class _TFMixin:
    def grads(self, x: tf.Tensor) -> tf.Tensor:
        grads, _ = self.grads_at(x)
        return grads

    def grads_at(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # We select again instead of using __call__
        #   to circumvent autograph's caveats of running
        #   Python side effects.
        fn = self._tf_function if self._use_tf else self._fn

        # Run fn with gradient tape on x
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = fn(x)

        return tape.gradient(y, x), y

    def enable_tf_function(self):
        self._use_tf = True

    def disable_tf_function(self):
        self._use_tf = False

    @cached_property
    def _domain_as_tensor(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return tuple(
            tf.constant(v, dtype=self._dtype)
            for v in [self.domain.min, self.domain.max]
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Evaluate the Tensorflow function with
        the passed tensor.

        Args:
            x (tf.Tensor): numeric tensor with shape (dims,)
                or (batch, dims).

        Returns:
            tf.Tensor: evaluation for x, either a tensor of
                shape () or (batch,)
        """
        #  Guarantee dtype
        x = tf.cast(x, dtype=self._dtype)

        # Select whether should be run eagerly or not
        fn = self._tf_function if self._use_tf else self._fn

        # Maybe check input shape
        if config.CHECK_INPUT_SHAPE:
            has_compatible_shape = len(x.shape) <= 2 and x.shape[-1] == self.dims
            if not has_compatible_shape:
                raise ValueError(
                    f"Incompatible shape for function {self.name} "
                    f"with {self.dims} dims: {x.shape}"
                )

        # Maybe check domain
        if config.CHECK_INPUT_DOMAIN:
            mi, ma = self._domain_as_tensor
            all_in_domain = tf.math.reduce_all(x >= mi) and tf.math.reduce_all(x <= ma)
            if not all_in_domain:
                if not config.COERCE_INPUT_TO_DOMAIN:
                    raise ValueError(
                        "Input values are out of bound for function "
                        f"{self.name} with domain {self.domain}."
                    )
                else:
                    x = tf.clip_by_value(x, mi, ma)

        return tf.cast(fn(x), dtype=self._dtype)


class TensorflowFunction(_TFMixin, core.Function):
    def __init__(
        self,
        dims: int,
        domain_min: Union[float, List[float]] = None,
        domain_max: Union[float, List[float]] = None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        if domain is None and domain_min is None:
            domain_min, domain_max = self.metadata.default_search_space
        super().__init__(dims, domain_min, domain_max, domain)
        self._use_tf = use_tf_function
        self._tf_function = tf.function(self._fn)
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Tensorflow", "")

    @property
    def metadata(self) -> core.Metadata:
        return FunctionMetadata[self.name]


class TensorflowTransformation(_TFMixin, core.Transformation):
    def __init__(
        self,
        fn: core.Function,
        vshift: float = 0.0,
        hshift: Union[float, List[float]] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: Union[float, List[float]] = 1.0,
        has_same_domain: bool = False,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        is_compatible = isinstance(fn, TensorflowFunction) or isinstance(
            fn, TensorflowTransformation
        )
        assert is_compatible, "Only TensorflowFunctions are accepted."
        super().__init__(fn, vshift, hshift, outer_scale, inner_scale, has_same_domain)
        self._use_tf = use_tf_function
        self._tf_function = tf.function(self._fn)
        self._dtype = dtype

    @cached_property
    def _params_as_tensor(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return tuple(
            tf.constant(p, dtype=self._dtype)
            for p in [self.vshift, self.hshift, self.outer_scale, self.inner_scale]
        )

    def _fn(self, x: tf.Tensor) -> tf.Tensor:
        # Get parameters
        vs, hs, os, iscale = self._params_as_tensor

        # Input transform
        x = iscale * x + hs

        # Apply function
        out = self.parent(x)

        # Apply output transforms
        out = os * out + vs

        return out


class AckleyTensorflow(TensorflowFunction):
    """Ackley function 1 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
        a: float = None,
        b: float = None,
        c: float = None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            use_tf_function=use_tf_function,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self.a = params["a"] if a is None else a
        self.b = params["b"] if a is None else b
        self.c = params["c"] if a is None else c

    def _fn(self, x: tf.Tensor):
        d = tf.constant(x.shape[-1], self._dtype)
        sum1 = tf.reduce_sum(tf.math.pow(x, 2), axis=-1)
        sum2 = tf.reduce_sum(tf.cos(tf.math.multiply(x, self.c)), axis=-1)
        term1 = tf.math.multiply(
            tf.exp(tf.math.multiply(tf.sqrt(tf.math.divide(sum1, d)), -self.b)), -self.a
        )
        term2 = tf.exp(tf.math.divide(sum2, d))
        result = term1 - term2 + self.a + e
        return result


class Alpine2Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.reduce_prod(tf.multiply(tf.sqrt(x), tf.sin(x)), axis=-1)


class BentCigarTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=1, limit=d, dtype=tf.int32)
        x0 = tf.squeeze(tf.gather(x, [0], axis=-1))
        xi = tf.gather(x, indices, axis=-1)
        return tf.pow(x0, 2) + tf.multiply(tf.reduce_sum(tf.pow(xi, 2), axis=-1), 1e6)


class BohachevskyTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        x0 = tf.squeeze(tf.gather(x, [0], axis=-1))
        x1 = tf.squeeze(tf.gather(x, [1], axis=-1))
        out = (
            tf.pow(x0, 2)
            + tf.math.multiply(tf.pow(x1, 2), 2)
            - tf.math.multiply(tf.cos(3 * pi * x0), 0.3)
            - tf.math.multiply(tf.cos(4 * pi * x1), 0.4)
            + 0.7
        )

        # Maybe batch with size 1?
        out = maybe_batch(out, tf.shape(x))

        return out


class BrownTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=0, limit=d, dtype=tf.int32)
        xi = tf.gather(x, indices[:-1], axis=-1)
        xi1 = tf.gather(x, indices[1:], axis=-1)

        xi_sq = tf.pow(xi, 2)
        xi1_sq = tf.pow(xi1, 2)

        return tf.reduce_sum(
            tf.pow(xi_sq, xi1_sq + 1) + tf.pow(xi1_sq, xi_sq + 1), axis=-1
        )


class ChungReynoldsTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), 2)


class CsendesTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.cond(
            tf.equal(tf.reduce_prod(x), 0),
            lambda: tf.reduce_sum(x * tf.constant(0, dtype=self._dtype), axis=-1),
            lambda: tf.reduce_sum(
                tf.multiply(tf.pow(x, 6), 2 + tf.sin(tf.divide(1, x))), axis=-1
            ),
        )


class Deb1Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(tf.pow(tf.sin(tf.multiply(x, 5 * pi)), 6), axis=-1), d
        )


class Deb3Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(
                tf.pow(tf.sin(tf.multiply(tf.pow(x, 3 / 4) - 0.05, 5 * pi)), 6), axis=-1
            ),
            d,
        )


class DixonPriceTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        x = atleast_2d(x)
        d = tf.shape(x)[-1]
        x0 = x[:, 0]
        ii = tf.range(2.0, d + 1, dtype=self._dtype)
        xi = x[:, 1:]
        xold = x[:, :-1]
        dixon_sum = ii * tf.pow(2 * tf.pow(xi, 2) - xold, 2)
        result = tf.pow(x0 - 1, 2) + tf.reduce_sum(dixon_sum, -1)
        result = maybe_batch(tf.squeeze(result), shape)
        return result


class ExponentialTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return -tf.exp(tf.multiply(tf.reduce_sum(tf.pow(x, 2), axis=-1), -0.5))


class GriewankTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        initial_shape = tf.shape(x)
        x = atleast_2d(x)
        shape = tf.shape(x)
        griewank_sum = tf.divide(tf.reduce_sum(tf.math.pow(x, 2), axis=-1), 4000)
        den = tf.range(1, shape[-1] + 1, dtype=self._dtype)
        den = tf.repeat(tf.expand_dims(den, 0), shape[0], axis=0)
        prod = tf.cos(tf.math.divide(x, tf.sqrt(den)))
        prod = tf.reduce_prod(prod, axis=-1)
        result = maybe_batch(tf.squeeze(griewank_sum - prod + 1), initial_shape)
        return result


class LevyTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        x = atleast_2d(x)
        d = tf.shape(x)[-1] - 1
        w = 1 + tf.math.divide(tf.math.subtract(x, 1), 4)

        term1 = tf.math.pow(tf.sin(pi * w[:, 0]), 2)
        wd = w[:, d]
        term3 = tf.math.pow(wd - 1, 2) * (1 + tf.math.pow(tf.sin(2 * pi * wd), 2))
        wi = w[:, 0:d]
        levy_sum = tf.reduce_sum(
            tf.math.pow((wi - 1), 2) * (1 + 10 * tf.math.pow(tf.sin(pi * wi + 1), 2)),
            axis=-1,
        )
        result = maybe_batch(tf.squeeze(term1 + levy_sum + term3), shape)
        return result


class Mishra2Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=0, limit=d, dtype=tf.int32)
        xi = tf.gather(x, indices[:-1], axis=-1)
        xi1 = tf.gather(x, indices[1:], axis=-1)
        xn = d - tf.reduce_sum(tf.multiply(xi + xi1, 0.5), axis=-1)
        return tf.pow(1 + xn, xn)


class PowellSumTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=1, limit=d + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.pow(tf.math.abs(x), indices + 1), axis=-1)


class QingTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=1, limit=d + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.pow(tf.pow(x, 2) - indices, 2), axis=-1)


class RastriginTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return (10 * d) + tf.reduce_sum(
            tf.math.pow(x, 2) - (10 * tf.cos(tf.math.multiply(x, 2 * pi))), axis=-1
        )


class RosenbrockTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        x = atleast_2d(x)
        xi = x[:, :-1]
        xnext = x[:, 1:]
        result = tf.reduce_sum(
            100 * tf.math.pow(xnext - tf.math.pow(xi, 2), 2) + tf.math.pow(xi - 1, 2),
            axis=-1,
        )
        result = maybe_batch(tf.squeeze(result), shape)
        return result


class RotatedHyperEllipsoidTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        x = atleast_2d(x)
        d = tf.shape(x)[-1]
        mat = tf.repeat(tf.expand_dims(x, 1), d, 1)
        matlow = tf.linalg.band_part(mat, -1, 0)
        inner = tf.reduce_sum(matlow**2, -1)
        result = tf.reduce_sum(inner, -1)
        result = maybe_batch(tf.squeeze(result), shape)
        return result


class SalomonTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        x_sqrt = tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
        return 1 - tf.cos(tf.multiply(x_sqrt, 2 * pi)) + tf.multiply(x_sqrt, 0.1)


class SarganTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        d = shape[-1]
        inner_sum_axis = tf.size(shape) - 1
        indices = tf.range(start=1, limit=d, dtype=tf.int32)
        has_batch = tf.math.logical_and(inner_sum_axis > 0, shape[0] > 1)
        xj = tf.expand_dims(tf.gather(x, indices, axis=-1), axis=-1)
        inner_x = tf.cond(
            has_batch, true_fn=lambda: tf.expand_dims(x, axis=1), false_fn=lambda: x
        )
        return tf.reduce_sum(
            tf.multiply(
                tf.cast(d, self._dtype),
                tf.pow(x, 2.0)
                + tf.multiply(tf.reduce_sum(inner_x * xj, axis=inner_sum_axis), 0.4),
            ),
            axis=-1,
        )


class SumSquaresTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        mul = tf.range(1, x.shape[-1] + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.math.multiply(tf.math.pow(x, 2), mul), axis=-1)


class SchumerSteiglitzTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.pow(x, 4), axis=-1)


class SchwefelTensorflow(TensorflowFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        a: float = None,
        dtype=tf.float32,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            use_tf_function=use_tf_function,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a

    def _fn(self, x: tf.Tensor):
        a = tf.cast(self._a, dtype=self._dtype)
        return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), a)


class Schwefel12Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.pow(tf.math.cumsum(x, axis=-1), 2), axis=-1)


class Schwefel222Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        x_abs = tf.abs(x)
        return tf.reduce_sum(x_abs, axis=-1) + tf.reduce_prod(x_abs, axis=-1)


class Schwefel223Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.pow(x, 10), axis=-1)


class Schwefel226Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(tf.multiply(x, tf.sin(tf.sqrt(tf.abs(x)))), axis=-1), d
        )


class SphereTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.math.pow(x, 2), axis=-1)


class StrechedVSineWaveTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=0, limit=d, dtype=tf.int32)
        xi_sqrd = tf.pow(tf.gather(x, indices[:-1], axis=-1), 2)
        xi1_sqrd = tf.pow(tf.gather(x, indices[1:], axis=-1), 2)
        sqrd_sum = xi1_sqrd + xi_sqrd
        return tf.reduce_sum(
            tf.multiply(
                tf.pow(sqrd_sum, 0.25),
                tf.pow(tf.sin(tf.multiply(tf.pow(sqrd_sum, 0.1), 50)), 2) + 0.1,
            ),
            axis=-1,
        )


class Trigonometric2Tensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        xi_squared = tf.pow(tf.subtract(x, 0.9), 2)
        x1_squared = tf.gather(xi_squared, [0], axis=-1)
        res_x = (
            tf.multiply(tf.pow(tf.sin(tf.multiply(xi_squared, 7)), 2), 8)
            + tf.multiply(tf.pow(tf.sin(tf.multiply(x1_squared, 14)), 2), 6)
            + xi_squared
        )
        return 1 + tf.reduce_sum(res_x, axis=-1)


class WWavyTensorflow(TensorflowFunction):
    def __init__(
        self,
        dims: int,
        domain_min=-pi,
        domain_max=pi,
        k: float = None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            use_tf_function=use_tf_function,
            dtype=dtype,
        )
        params = self.metadata.default_parameters
        self._k = params["k"] if k is None else k

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return 1 - tf.divide(
            tf.reduce_sum(
                tf.multiply(
                    tf.cos(tf.multiply(x, self._k)), tf.exp(tf.divide(-tf.pow(x, 2), 2))
                ),
                axis=-1,
            ),
            d,
        )


class WeierstrassTensorflow(TensorflowFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
        a: float = None,
        b: float = None,
        kmax: int = None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            use_tf_function=use_tf_function,
            dtype=dtype,
        )
        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a
        self._b = params["b"] if a is None else b
        self._kmax = params["kmax"] if a is None else kmax

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        kindices = tf.range(start=0, limit=self._kmax + 1, dtype=self._dtype)

        #  Constants
        ak = tf.pow(self._a, kindices)
        bk = tf.pow(self._b, kindices)
        ak_cos_pi_bk = d * tf.reduce_sum(
            tf.multiply(ak, tf.cos(tf.multiply(bk, pi))), axis=-1
        )

        # Inner x
        inner_x = tf.expand_dims(tf.add(x, 0.5), axis=-1)
        ak_cos_2pibk = tf.reduce_sum(ak * tf.cos(2 * pi * bk * inner_x), axis=-1)
        return tf.reduce_sum(ak_cos_2pibk, axis=-1) - ak_cos_pi_bk


class WhitleyTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        shape = tf.shape(x)
        d = shape[-1]
        has_batch = tf.size(shape) > 1
        initializer = tf.cond(
            has_batch,
            true_fn=lambda: tf.zeros((shape[0],), dtype=self._dtype),
            false_fn=lambda: tf.cast(0.0, dtype=self._dtype),
        )
        indices = tf.range(start=0, limit=d, dtype=tf.int32)

        def fn(acc, j):
            # xi, xj
            xi = x
            xj = tf.gather(x, j, axis=-1)
            xj = tf.cond(
                has_batch,
                true_fn=lambda: tf.expand_dims(xj, axis=-1),
                false_fn=lambda: xj,
            )

            # Terms
            xi_sqrd = tf.pow(xi, 2)
            xi_sqrd_minus_xj_all_sqrd = tf.pow(tf.subtract(xi_sqrd, xj), 2)
            one_minus_xj_all_sqrd = tf.pow(-tf.subtract(xj, 1), 2)
            hundred_composite = tf.multiply(xi_sqrd_minus_xj_all_sqrd, 100)

            # Terms
            t1 = tf.divide(
                tf.pow(
                    tf.add(
                        hundred_composite,
                        one_minus_xj_all_sqrd,
                    ),
                    2,
                ),
                4000,
            )
            t2 = tf.cos(
                tf.add(hundred_composite, one_minus_xj_all_sqrd),
            )

            # Accumulated value
            return acc + tf.reduce_sum(t1 - t2 + 1, axis=-1)

        return tf.foldl(fn, indices, initializer=initializer)


class ZakharovTensorflow(TensorflowFunction):
    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        sum1 = tf.reduce_sum(tf.math.pow(x, 2), axis=-1)
        sum2 = tf.reduce_sum(
            tf.math.divide(
                tf.math.multiply(x, tf.range(1, (d + 1), dtype=self._dtype)), 2
            ),
            axis=-1,
        )
        return sum1 + tf.math.pow(sum2, 2) + tf.math.pow(sum2, 4)


def atleast_2d(tensor: tf.Tensor) -> tf.Tensor:
    """Make sure a tensor is a matrix."""
    return tf.cond(
        tf.less(tf.size(tf.shape(tensor)), 2),
        lambda: tf.expand_dims(tensor, 0),
        lambda: tensor,
    )


def maybe_batch(value: tf.Tensor, shape) -> tf.Tensor:
    return tf.cond(
        tf.math.logical_and(tf.size(shape) > 1, shape[0] == 1),
        true_fn=lambda: tf.expand_dims(value, axis=0),
        false_fn=lambda: value,
    )
