"""TensorFlow's implementation of many functions.
References:
  [1] Momin Jamil and Xin-She Yang.
      A Literature Survey of Benchmark Functions For Global
        Optimization Problems, 2013. (10.1504/IJMMNO.2013.055204)
  [2] IEEE CEC 2021 C-2 (https://cec2021.mini.pw.edu.pl/en/program/competitions)
  [3] IEEE CEC 2021 C-3 (https://cec2021.mini.pw.edu.pl/en/program/competitions)
  [4] https://www.sfu.ca/~ssurjano/optimization.html
"""

from functools import cached_property
from math import e, pi

import tensorflow as tf

from py_benchmark_functions import config, core
from py_benchmark_functions.info import FunctionMetadata

# Ensure tf.function's are run as graphs
tf.config.run_functions_eagerly(False)


class _TFMixin:
    def grads(self, x: tf.Tensor) -> tf.Tensor:
        grads, _ = self.grads_at(x)
        return grads

    def grads_at(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
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
    def _domain_as_tensor(self) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.constant(self.domain.min), tf.constant(self.domain.max)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Evaluate the Tensorflow function with
        the passed tensor.

        Args:
            x (tf.Tensor): numeric tensor with shape (dims,)
                or (batch, dims).

        Returns:
            tf.Tensor: evaluation for x, either a tensor of
                shape (1,) or (batch,)
        """
        #  Guarantee dtype
        x = tf.cast(x, dytpe=self._dtype)

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


class TensorflowFunction(core.Function, _TFMixin):
    def __init__(
        self,
        dims: int,
        domain_min: float | list[float] = None,
        domain_max: float | list[float] = None,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
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


class TensorflowTransformation(core.Transformation, _TFMixin):
    def __init__(
        self,
        fn: core.Function,
        vshift: float = 0.0,
        hshift: float | list[float] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: float | list[float] = 1.0,
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
    def _params_as_tensor(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return tuple(
            tf.constant(p)
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
        domain_min=-32.768,
        domain_max=32.768,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
        a=20,
        b=0.2,
        c=2 * pi,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)
        self.a = a
        self.b = b
        self.c = c

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
    """Alpine function 2 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=0.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.reduce_prod(tf.multiply(tf.sqrt(x), tf.sin(x)), axis=-1)


class BentCigarTensorflow(TensorflowFunction):
    """BentCigar function defined in [2].
    Implementation doesn't support batch yet.
    """

    def __init__(
        self,
        dims: int,
        domain_min=0.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.pow(x[0], 2) + tf.multiply(
            tf.reduce_sum(tf.pow(x[1:], 2), axis=-1), 1e6
        )


class BohachevskyTensorflow(TensorflowFunction):
    """Bohachevsky function 1 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        assert dims == 2, "Bohachevsky only supports 2 dimensions."
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return (
            tf.pow(x[0], 2)
            + tf.math.multiply(tf.pow(x[1], 2), 2)
            - tf.math.multiply(tf.cos(3 * pi * x[0]), 0.3)
            - tf.math.multiply(tf.cos(4 * pi * x[1]), 0.4)
            + 0.7
        )


class BrownTensorflow(TensorflowFunction):
    """Brown function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-1.0,
        domain_max=4.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        xi = x[: d - 1]  # len = d-1
        xi1 = x[1:d]  # len = d-1

        xi_sq = tf.pow(xi, 2)
        xi1_sq = tf.pow(xi1, 2)

        return tf.reduce_sum(
            tf.pow(xi_sq, xi1_sq + 1) + tf.pow(xi1_sq, xi_sq + 1), axis=-1
        )


class ChungReynoldsTensorflow(TensorflowFunction):
    """Chung Reynolds function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), 2)


class CsendesTensorflow(TensorflowFunction):
    """Csendes function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-1.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.cond(
            tf.equal(tf.reduce_prod(x), 0),
            lambda: tf.reduce_sum(x * tf.constant(0, dtype=self._dtype), axis=-1),
            lambda: tf.reduce_sum(
                tf.multiply(tf.pow(x, 6), 2 + tf.sin(tf.divide(1, x))), axis=-1
            ),
        )


class Deb1Tensorflow(TensorflowFunction):
    """Deb function 1 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-1.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(tf.pow(tf.sin(tf.multiply(x, 5 * pi)), 6), axis=-1), d
        )


class Deb3Tensorflow(TensorflowFunction):
    """Deb function 3 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=0.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(
                tf.pow(tf.sin(tf.multiply(tf.pow(x, 3 / 4) - 0.05, 5 * pi)), 6), axis=-1
            ),
            d,
        )


class DixonPriceTensorflow(TensorflowFunction):
    """Dixon-Price function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x = atleast_2d(x)
        d = tf.shape(x)[-1]
        x0 = x[:, 0]
        ii = tf.range(2.0, d + 1, dtype=self._dtype)
        xi = x[:, 1:]
        xold = x[:, :-1]
        dixon_sum = ii * tf.pow(2 * tf.pow(xi, 2) - xold, 2)
        result = tf.pow(x0 - 1, 2) + tf.reduce_sum(dixon_sum, -1)
        return tf.squeeze(result)


class ExponentialTensorflow(TensorflowFunction):
    """Exponential function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-1.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return -tf.exp(tf.multiply(tf.reduce_sum(tf.pow(x, 2), axis=-1), -0.5))


class GriewankTensorflow(TensorflowFunction):
    """Griewank function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x = atleast_2d(x)
        shape = tf.shape(x)
        griewank_sum = tf.divide(tf.reduce_sum(tf.math.pow(x, 2), axis=-1), 4000)
        den = tf.range(1, shape[-1] + 1, dtype=self._dtype)
        den = tf.repeat(tf.expand_dims(den, 0), shape[0], axis=0)
        prod = tf.cos(tf.math.divide(x, tf.sqrt(den)))
        prod = tf.reduce_prod(prod, axis=-1)
        return tf.squeeze(griewank_sum - prod + 1)


class LevyTensorflow(TensorflowFunction):
    """Levy function defined in [4]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
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
        return tf.squeeze(term1 + levy_sum + term3)


class Mishra2Tensorflow(TensorflowFunction):
    """Mishra function 2 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=0.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        xi = x[: d - 1]
        xi1 = x[1:]
        xn = d - tf.reduce_sum(tf.multiply(xi + xi1, 0.5), axis=-1)
        return tf.pow(1 + xn, xn)


class PowellSumTensorflow(TensorflowFunction):
    """Powell Sum function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-1.0,
        domain_max=1.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=1, limit=d + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.pow(tf.math.abs(x), indices + 1))


class QingTensorflow(TensorflowFunction):
    """Qing function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-500.0,
        domain_max=500.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=1, limit=d + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.pow(tf.pow(x, 2) - indices, 2), axis=-1)


class RastriginTensorflow(TensorflowFunction):
    """Rastrigin function defined in [2]. Search range may vary."""

    def __init__(
        self,
        dims: int,
        domain_min=-5.12,
        domain_max=5.12,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return (10 * d) + tf.reduce_sum(
            tf.math.pow(x, 2) - (10 * tf.cos(tf.math.multiply(x, 2 * pi))), axis=-1
        )


class RosenbrockTensorflow(TensorflowFunction):
    """Rosenbrock function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-30.0,
        domain_max=30.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x = atleast_2d(x)
        xi = x[:, :-1]
        xnext = x[:, 1:]
        result = tf.reduce_sum(
            100 * tf.math.pow(xnext - tf.math.pow(xi, 2), 2) + tf.math.pow(xi - 1, 2),
            axis=-1,
        )
        return tf.squeeze(result)


class RotatedHyperEllipsoidTensorflow(TensorflowFunction):
    """Rotated Hyper-Ellipsoid function defined in [4]."""

    def __init__(
        self,
        dims: int,
        domain_min=-65.536,
        domain_max=65.536,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x = atleast_2d(x)
        d = tf.shape(x)[-1]
        mat = tf.repeat(tf.expand_dims(x, 1), d, 1)
        matlow = tf.linalg.band_part(mat, -1, 0)
        inner = tf.reduce_sum(matlow**2, -1)
        result = tf.reduce_sum(inner, -1)
        return tf.squeeze(result)


class SalomonTensorflow(TensorflowFunction):
    """Salomon function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x_sqrt = tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
        return 1 - tf.cos(tf.multiply(x_sqrt, 2 * pi)) + tf.multiply(x_sqrt, 0.1)


class SarganTensorflow(TensorflowFunction):
    """Sargan function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        xj = x[1:]

        def fn(acc, xi):
            return acc + tf.multiply(
                tf.pow(xi, 2)
                + tf.multiply(tf.reduce_sum(tf.multiply(xj, xi), axis=-1), 0.4),
                d,
            )

        return tf.foldl(fn, x, initializer=tf.cast(0, dtype=self._dtype))


class SumSquaresTensorflow(TensorflowFunction):
    """Sum Squares function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        mul = tf.range(1, x.shape[-1] + 1, dtype=self._dtype)
        return tf.reduce_sum(tf.math.multiply(tf.math.pow(x, 2), mul), axis=-1)


class SchwefelTensorflow(TensorflowFunction):
    """Schwefel function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        a: float = pi,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)
        self._a = a

    def _fn(self, x: tf.Tensor):
        a = tf.cast(self._a, dtype=self._dtype)
        return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), a)


class Schwefel12Tensorflow(TensorflowFunction):
    """Schwefel function 1.2 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=0, limit=d, dtype=tf.int32)

        def fn(acc, i):
            gather_indices = tf.range(start=0, limit=i + 1, dtype=tf.int32)
            x_ = tf.gather(x, gather_indices)
            return acc + tf.pow(tf.reduce_sum(x_, axis=-1), 2)

        return tf.foldl(fn, indices, initializer=tf.cast(0, dtype=self._dtype))


class Schwefel222Tensorflow(TensorflowFunction):
    """Schwefel function 2.22 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-100.0,
        domain_max=100.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        x_abs = tf.abs(x)
        return tf.reduce_sum(x_abs, axis=-1) + tf.reduce_prod(x_abs, axis=-1)


class Schwefel223Tensorflow(TensorflowFunction):
    """Schwefel function 2.23 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.pow(x, 10), axis=-1)


class Schwefel226Tensorflow(TensorflowFunction):
    """Schwefel function 2.26 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-500.0,
        domain_max=500.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        return -tf.divide(
            tf.reduce_sum(tf.multiply(x, tf.sin(tf.sqrt(tf.abs(x)))), axis=-1), d
        )


class SchumerSteiglitzTensorflow(TensorflowFunction):
    """Schumer Steiglitz function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.pow(x, 4), axis=-1)


class SphereTensorflow(TensorflowFunction):
    """Sphere function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=0.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        return tf.reduce_sum(tf.math.pow(x, 2), axis=-1)


class StrechedVSineWaveTensorflow(TensorflowFunction):
    """Streched V Sine Wave function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        xi_sqrd = tf.pow(x[: d - 1], 2)
        xi1_sqrd = tf.pow(x[1:], 2)
        sqrd_sum = xi_sqrd + xi1_sqrd

        return tf.reduce_sum(
            tf.multiply(
                tf.pow(sqrd_sum, 0.25),
                tf.pow(tf.sin(tf.multiply(tf.pow(sqrd_sum, 0.1), 50)), 2) + 0.1,
            ),
            axis=-1,
        )


class Trigonometric2Tensorflow(TensorflowFunction):
    """Trigonometric function 2 defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-500.0,
        domain_max=500.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        xi_squared = tf.pow(tf.subtract(x, 0.9), 2)

        res_x = (
            tf.multiply(tf.pow(tf.sin(tf.multiply(xi_squared, 7)), 2), 8)
            + tf.multiply(tf.pow(tf.sin(tf.multiply(xi_squared, 14)), 2), 6)
            + xi_squared
        )
        return 1 + tf.reduce_sum(res_x, axis=-1)


class WeierstrassTensorflow(TensorflowFunction):
    """Weierstrass function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-0.5,
        domain_max=0.5,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
        a: float = 0.5,
        b: float = 3,
        kmax: int = 20,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)
        self._a = a
        self._b = b
        self._kmax = kmax

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        a = tf.cast(self._a, dtype=self._dtype)
        b = tf.cast(self._b, dtype=self._dtype)

        kindices = tf.range(start=0, limit=self._kmax + 1, dtype=self._dtype)
        ak = tf.pow(a, kindices)
        bk = tf.pow(b, kindices)

        ak_bk_sum = d * tf.reduce_sum(
            tf.multiply(ak, tf.cos(tf.multiply(bk, pi))), axis=-1
        )

        def fn(acc, xi):
            s = tf.reduce_sum(
                tf.multiply(ak, tf.cos(tf.multiply(2 * pi * bk, xi + 0.5))), axis=-1
            )
            return acc + (s - ak_bk_sum)

        return tf.foldl(fn, x, initializer=tf.cast(0, dtype=self._dtype))


class WhitleyTensorflow(TensorflowFunction):
    """Whitley function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-10.24,
        domain_max=10.24,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

    def _fn(self, x: tf.Tensor):
        d = x.shape[-1]
        indices = tf.range(start=0, limit=d, dtype=tf.int32)

        def fn(acc, i):
            xi_sqrd = tf.pow(tf.gather(x, i), 2)
            ij_diff_sqrd = tf.pow(tf.subtract(xi_sqrd, x), 2)
            aux = tf.pow(-tf.subtract(x, 1), 2)
            t1 = tf.divide(tf.pow(tf.multiply(ij_diff_sqrd, 100) + aux, 2), 4000)
            t2 = tf.cos(tf.multiply(ij_diff_sqrd, 100) + aux)
            return acc + tf.reduce_sum(t1 - t2 + 1, axis=-1)

        return tf.foldl(fn, indices, initializer=tf.cast(0, dtype=self._dtype))


class WWavyTensorflow(TensorflowFunction):
    """W / Wavy function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-pi,
        domain_max=pi,
        k: float = 10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)
        self._k = k

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


class ZakharovTensorflow(TensorflowFunction):
    """Zakharov function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-5.0,
        domain_max=10.0,
        domain: core.Domain = None,
        use_tf_function: bool = True,
        dtype=tf.float32,
    ):
        super().__init__(dims, domain_min, domain_max, domain, use_tf_function, dtype)

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
