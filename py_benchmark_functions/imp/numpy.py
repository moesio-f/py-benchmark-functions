"""NumPy's implementation of many functions.

References:
  [1] https://www.sfu.ca/~ssurjano/optimization.html
"""

from functools import cached_property
from typing import List, Tuple, Union

import numpy as np
from numpy import pi

from py_benchmark_functions import config, core
from py_benchmark_functions.info import FunctionMetadata


class _NPMixin:
    def grads(self, x: np.ndarray):
        raise NotImplementedError("Gradients for NumPy functions are not supported.")

    def grads_at(self, x: np.ndarray):
        raise NotImplementedError("Gradients for NumPy functions are not supported.")

    @cached_property
    def _domain_as_array(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.domain.min), np.array(self.domain.max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the numpy function with the
        passed numpy array.

        Args:
            x (np.ndarray): numeric array with shape (dims,) or
                (batch, dims).

        Returns:
            np.ndarray: evaluation for x, either an array of
                shape () or (batch,).
        """
        # Guarantee expected dtype
        x = x.astype(self._dtype)

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
            mi, ma = self._domain_as_array
            all_in_domain = (x >= mi).all() and (x <= ma).all()
            if not all_in_domain:
                if not config.COERCE_INPUT_TO_DOMAIN:
                    raise ValueError(
                        "Input values are out of bound for function "
                        f"{self.name} with domain {self.domain}."
                    )
                else:
                    x = np.clip(x, mi, ma)

        # Run function
        result = self._fn(x).astype(self._dtype)
        return result


class NumpyFunction(_NPMixin, core.Function):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain=None,
        dtype=np.float32,
    ):
        if domain is None and domain_min is None:
            domain_min, domain_max = self.metadata.default_search_space
        super().__init__(dims, domain_min, domain_max, domain)
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Numpy", "")

    @property
    def metadata(self) -> core.Metadata:
        return FunctionMetadata[self.name]


class NumpyTransformation(_NPMixin, core.Transformation):
    def __init__(
        self,
        fn: core.Function,
        vshift: float = 0.0,
        hshift: Union[float, List[float]] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: Union[float, List[float]] = 1.0,
        has_same_domain: bool = False,
        dtype=np.float32,
    ):
        is_compatible = isinstance(fn, NumpyFunction) or isinstance(
            fn, NumpyTransformation
        )
        assert is_compatible, "Only NumpyFunctions are accepted."
        super().__init__(fn, vshift, hshift, outer_scale, inner_scale, has_same_domain)
        self._dtype = dtype

    @cached_property
    def _params_as_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.array(p, dtype=self._dtype)
            for p in [self.vshift, self.hshift, self.outer_scale, self.inner_scale]
        )

    def _fn(self, x: np.ndarray) -> np.ndarray:
        # Get parameters
        vs, hs, os, iscale = self._params_as_array

        # Input transform
        x = iscale * x + hs

        # Apply function
        out = self.parent(x)

        # Apply output transforms
        out = os * out + vs

        return out


class AckleyNumpy(NumpyFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain=None,
        a: float = None,
        b: float = None,
        c: float = None,
        dtype=np.float32,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self.a = params["a"] if a is None else a
        self.b = params["b"] if a is None else b
        self.c = params["c"] if a is None else c

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(np.cos(self.c * x), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.e

        return result


class Alpine2Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return np.prod(np.sqrt(x) * np.sin(x), axis=-1)


class BentCigarNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=1, stop=d, dtype=np.int32)
        x0 = np.take(x, [0], axis=-1).squeeze()
        xi = np.take(x, indices, axis=-1)
        return np.power(x0, 2) + (np.sum(np.power(xi, 2), axis=-1) * 1e6)


class BohachevskyNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        x0 = np.take(x, 0, axis=-1)
        x1 = np.take(x, 1, axis=-1)
        result = (
            np.power(x0, 2)
            + 2 * np.power(x1, 2)
            - 0.3 * np.cos(3 * np.pi * x0)
            - 0.4 * np.cos(4 * np.pi * x1)
            + 0.7
        )
        return result


class BrownNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=0, stop=d, dtype=np.int32)
        xi = np.take(x, indices[:-1], axis=-1)
        xi1 = np.take(x, indices[1:], axis=-1)
        xi_sq = np.power(xi, 2)
        xi1_sq = np.power(xi1, 2)
        return np.sum(np.power(xi_sq, xi1_sq + 1) + np.power(xi1_sq, xi_sq + 1), axis=-1)


class ChungReynoldsNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return np.power(np.sum(np.power(x, 2), axis=-1), 2)


class CsendesNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        if np.prod(x) == 0.0:
            return np.sum(x * 0, axis=-1)

        return np.sum((np.power(x, 6)) * (2 + np.sin(1 / x)), axis=-1)


class Deb1Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        return -np.divide(np.sum(np.power(np.sin(np.multiply(x, 5 * pi)), 6), axis=-1), d)


class Deb3Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        return -np.divide(
            np.sum(
                np.power(np.sin(np.multiply(np.power(x, 3 / 4) - 0.05, 5 * pi)), 6), axis=-1
            ),
            d,
        )


class DixonPriceNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        initial_shape = x.shape
        x = np.atleast_2d(x)
        x0 = x[:, 0]
        d = x.shape[-1]
        ii = np.arange(2.0, d + 1)
        xi = x[:, 1:]
        xold = x[:, :-1]
        dixon_sum = ii * (2 * xi**2 - xold) ** 2
        result = maybe_squeeze(
            (x0 - 1) ** 2 + np.sum(dixon_sum, axis=-1), initial_shape
        )
        return result


class ExponentialNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return -np.exp(np.multiply(np.sum(np.power(x, 2), axis=-1), -0.5))


class GriewankNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        initial_shape = x.shape
        x = np.atleast_2d(x)
        griewank_sum = np.sum(x**2, axis=-1) / 4000.0
        den = np.arange(1, x.shape[-1] + 1, dtype=x.dtype)[None].repeat(
            x.shape[0], axis=0
        )
        prod = np.cos(x / np.sqrt(den))
        prod = np.prod(prod, axis=-1)
        result = maybe_squeeze(griewank_sum - prod + 1, initial_shape)
        return result


class LevyNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        initial_shape = x.shape
        x = np.atleast_2d(x)
        pi = np.pi
        d = x.shape[-1] - 1
        w = 1 + (x - 1) / 4
        term1 = np.sin(pi * w[:, 0]) ** 2
        wd = w[:, d]
        term3 = (wd - 1) ** 2 * (1 + np.sin(2 * pi * wd) ** 2)
        wi = w[:, 0:d]
        levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2), axis=-1)
        result = maybe_squeeze(term1 + levy_sum + term3, initial_shape)
        return result


class Mishra2Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=0, stop=d, dtype=np.int32)
        xi = np.take(x, indices[:-1], axis=-1)
        xi1 = np.take(x, indices[1:], axis=-1)
        xn = d - np.sum(np.multiply(xi + xi1, 0.5), axis=-1)
        return np.power(1 + xn, xn)


class PowellSumNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=1, stop=d + 1, dtype=self._dtype)
        return np.sum(np.power(np.abs(x), indices + 1), axis=-1)


class QingNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=1, stop=d + 1, dtype=self._dtype)
        return np.sum(np.power(np.power(x, 2) - indices, 2), axis=-1)


class RastriginNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        result = 10 * d + np.sum(x**2 - 10 * np.cos(x * 2 * np.pi), axis=-1)
        return result


class RosenbrockNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        initial_shape = x.shape
        x = np.atleast_2d(x)
        xi = x[:, :-1]
        xnext = x[:, 1:]
        result = maybe_squeeze(
            np.sum(100 * (xnext - xi**2) ** 2 + (xi - 1) ** 2, axis=-1), initial_shape
        )
        return result


class RotatedHyperEllipsoidNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        initial_shape = x.shape
        x = np.atleast_2d(x)
        mat = x[:, None].repeat(x.shape[-1], axis=1)
        matlow = np.tril(mat)
        inner = np.sum(matlow**2, axis=-1)
        result = maybe_squeeze(np.sum(inner, axis=-1), initial_shape)
        return result


class SalomonNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        x_sqrt = np.sqrt(np.sum(np.power(x, 2), axis=-1))
        return 1 - np.cos(np.multiply(x_sqrt, 2 * pi)) + np.multiply(x_sqrt, 0.1)


class SarganNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        shape = np.shape(x)
        d = shape[-1]
        inner_sum_axis = np.size(shape) - 1
        indices = np.arange(start=1, stop=d, dtype=np.int32)
        has_batch = np.logical_and(inner_sum_axis > 0, shape[0] > 1)

        xj = np.expand_dims(np.take(x, indices, axis=-1), axis=-1)
        inner_x = np.expand_dims(x, axis=1) if has_batch else x
        return np.sum(
            np.multiply(
                d,
                np.power(x, 2.0)
                + np.multiply(np.sum(inner_x * xj, axis=inner_sum_axis), 0.4),
            ),
            axis=-1,
        )


class SumSquaresNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
        result = np.sum((x**2) * mul, axis=-1)

        return result


class SchumerSteiglitzNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return np.sum(np.power(x, 4), axis=-1)


class SchwefelNumpy(NumpyFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        a: float = None,
        dtype=np.float32,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a

    def _fn(self, x: np.ndarray):
        return np.power(np.sum(np.power(x, 2), axis=-1), self._a)


class Schwefel12Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return np.sum(np.power(np.cumsum(x, axis=-1), 2), axis=-1)


class Schwefel222Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        x_abs = np.abs(x)
        return np.sum(x_abs, axis=-1) + np.prod(x_abs, axis=-1)


class Schwefel223Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        return np.sum(np.power(x, 10), axis=-1)


class Schwefel226Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        return -np.divide(
            np.sum(np.multiply(x, np.sin(np.sqrt(np.abs(x)))), axis=-1), d
        )


class SphereNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        result = np.sum(x * x, axis=-1)
        return result


class StrechedVSineWaveNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        indices = np.arange(start=0, stop=d, dtype=np.int32)
        xi_sqrd = np.power(np.take(x, indices[:-1], axis=-1), 2)
        xi1_sqrd = np.power(np.take(x, indices[1:], axis=-1), 2)
        sqrd_sum = xi1_sqrd + xi_sqrd

        return np.sum(
            np.multiply(
                np.power(sqrd_sum, 0.25),
                np.power(np.sin(np.multiply(np.power(sqrd_sum, 0.1), 50)), 2) + 0.1,
            ),
            axis=-1,
        )


class Trigonometric2Numpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        xi_squared = np.power(np.subtract(x, 0.9), 2)
        x1_squared = np.take(xi_squared, [0], axis=-1)

        res_x = (
            np.multiply(np.power(np.sin(np.multiply(xi_squared, 7)), 2), 8)
            + np.multiply(np.power(np.sin(np.multiply(x1_squared, 14)), 2), 6)
            + xi_squared
        )
        return 1 + np.sum(res_x, axis=-1)


class WWavyNumpy(NumpyFunction):
    def __init__(
        self,
        dims: int,
        domain_min=-pi,
        domain_max=pi,
        k: float = None,
        domain: core.Domain = None,
        dtype=np.float32,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self._k = params["k"] if k is None else k

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        return 1 - np.divide(
            np.sum(
                np.multiply(
                    np.cos(np.multiply(x, self._k)), np.exp(np.divide(-np.power(x, 2), 2))
                ),
                axis=-1,
            ),
            d,
        )


class WeierstrassNumpy(NumpyFunction):
    def __init__(
        self,
        dims: int,
        domain_min=None,
        domain_max=None,
        domain: core.Domain = None,
        dtype=np.float32,
        a: float = None,
        b: float = None,
        kmax: int = None,
    ):
        super().__init__(
            dims,
            domain_min=domain_min,
            domain_max=domain_max,
            domain=domain,
            dtype=dtype,
        )

        params = self.metadata.default_parameters
        self._a = params["a"] if a is None else a
        self._b = params["b"] if a is None else b
        self._kmax = params["kmax"] if a is None else kmax

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        kindices = np.arange(start=0, stop=self._kmax + 1, dtype=self._dtype)

        #  Constants
        ak = np.power(self._a, kindices)
        bk = np.power(self._b, kindices)
        ak_cos_pi_bk = d * np.sum(np.multiply(ak, np.cos(np.multiply(bk, pi))), axis=-1)

        # Inner x
        inner_x = np.expand_dims(np.add(x, 0.5), axis=-1)
        ak_cos_2pibk = np.sum(ak * np.cos(2 * pi * bk * inner_x), axis=-1)
        return np.sum(ak_cos_2pibk, axis=-1) - ak_cos_pi_bk


class WhitleyNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        shape = np.shape(x)
        d = shape[-1]
        has_batch = np.size(shape) > 1
        indices = np.arange(start=0, stop=d, dtype=np.int32)
        if has_batch:
            acc = np.zeros((shape[0],), dtype=self._dtype)
        else:
            acc = np.array(0.0, dtype=self._dtype)

        for j in indices:
            xj = np.take(x, j, axis=-1)
            if has_batch:
                xj = np.expand_dims(xj, axis=-1)

            # Terms
            xi_sqrd = np.power(x, 2)
            xi_sqrd_minus_xj_all_sqrd = np.power(np.subtract(xi_sqrd, xj), 2)
            one_minus_xj_all_sqrd = np.power(-np.subtract(xj, 1), 2)
            hundred_composite = np.multiply(xi_sqrd_minus_xj_all_sqrd, 100)

            # Terms
            t1 = np.divide(
                np.power(
                    np.add(
                        hundred_composite,
                        one_minus_xj_all_sqrd,
                    ),
                    2,
                ),
                4000,
            )
            t2 = np.cos(
                np.add(hundred_composite, one_minus_xj_all_sqrd),
            )

            # Accumulated value
            acc += np.sum(t1 - t2 + 1, axis=-1)

        return acc


class ZakharovNumpy(NumpyFunction):
    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(x * np.arange(start=1, stop=(d + 1), dtype=x.dtype) / 2, axis=-1)
        result = sum1 + sum2**2 + sum2**4
        return result


def maybe_squeeze(value: np.ndarray, initial_shape: tuple):
    # If initial shape wasn't (batch, d) where batch > 1,
    #   then it should be squeezed to (1,)
    if len(initial_shape) <= 1 or initial_shape[0] != 1:
        value = np.squeeze(value)

    return value
