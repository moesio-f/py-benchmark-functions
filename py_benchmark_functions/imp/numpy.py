"""NumPy's implementation of many functions.

References:
  [1] https://www.sfu.ca/~ssurjano/optimization.html
"""

from functools import cached_property

import numpy as np

from py_benchmark_functions import config, core
from py_benchmark_functions.info import FunctionMetadata


class _NPMixin:
    def grads(self, x: np.ndarray):
        raise NotImplementedError("Gradients for NumPy functions are not supported.")

    def grads_at(self, x: np.ndarray):
        raise NotImplementedError("Gradients for NumPy functions are not supported.")

    @cached_property
    def _domain_as_array(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.domain.min), np.array(self.domain.max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the numpy function with the
        passed numpy array.

        Args:
            x (np.ndarray): numeric array with shape (dims,) or
                (batch, dims).

        Returns:
            np.ndarray: evaluation for x, either an array of
                shape (1,) or (batch,).
        """
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
        result = self._fn(x)

        # Maybe different dtype?
        if result.dtype != x.dtype:
            result = result.astype(x.dtype)

        return result


class NumpyFunction(core.Function, _NPMixin):
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Numpy", "")

    @property
    def metadata(self) -> core.Metadata:
        return FunctionMetadata[self.name]


class NumpyTransformation(core.Transformation, _NPMixin):
    def __init__(
        self,
        fn: core.Function,
        vshift: float = 0.0,
        hshift: float | list[float] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: float | list[float] = 1.0,
        has_same_domain: bool = False,
    ):
        is_compatible = isinstance(fn, NumpyFunction) or isinstance(
            fn, NumpyTransformation
        )
        assert is_compatible, "Only NumpyFunctions are accepted."
        super().__init__(fn, vshift, hshift, outer_scale, inner_scale, has_same_domain)

    @cached_property
    def _params_as_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.array(p)
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
    """Ackley function defined in [1]."""

    def __init__(
        self,
        dims: int,
        domain_min=-32.768,
        domain_max=32.768,
        a=20,
        b=0.2,
        c=2 * np.pi,
        dtype=np.float32,
    ):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)
        self.a = a
        self.b = b
        self.c = c
        self.dtype = dtype

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(np.cos(self.c * x), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.e

        return result


class GriewankNumpy(NumpyFunction):
    """Griewank function defined in [1]."""

    def __init__(self, dims: int, domain_min=-600.0, domain_max=600.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        x = np.atleast_2d(x)
        griewank_sum = np.sum(x**2, axis=-1) / 4000.0
        den = np.arange(1, x.shape[-1] + 1, dtype=x.dtype)[None].repeat(
            x.shape[0], axis=0
        )
        prod = np.cos(x / np.sqrt(den))
        prod = np.prod(prod, axis=-1)
        result = griewank_sum - prod + 1

        return np.squeeze(result)


class RastriginNumpy(NumpyFunction):
    """Rastrigin function defined in [1]."""

    def __init__(self, dims: int, domain_min=-5.12, domain_max=5.12):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        result = 10 * d + np.sum(x**2 - 10 * np.cos(x * 2 * np.pi), axis=-1)

        return result


class LevyNumpy(NumpyFunction):
    """Levy function defined in [1]."""

    def __init__(self, dims: int, domain_min=-10.0, domain_max=10.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        x = np.atleast_2d(x)
        pi = np.pi
        d = x.shape[-1] - 1
        w = 1 + (x - 1) / 4

        term1 = np.sin(pi * w[:, 0]) ** 2
        wd = w[:, d]
        term3 = (wd - 1) ** 2 * (1 + np.sin(2 * pi * wd) ** 2)
        wi = w[:, 0:d]
        levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2), axis=-1)
        result = term1 + levy_sum + term3

        return np.squeeze(result)


class RosenbrockNumpy(NumpyFunction):
    """Rosenbrock function defined in [1]."""

    def __init__(self, dims: int, domain_min=-5.0, domain_max=10.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        x = np.atleast_2d(x)
        xi = x[:, :-1]
        xnext = x[:, 1:]
        result = np.sum(100 * (xnext - xi**2) ** 2 + (xi - 1) ** 2, axis=-1)

        return np.squeeze(result)


class ZakharovNumpy(NumpyFunction):
    """Zakharov function defined in [1]."""

    def __init__(self, dims: int, domain_min=-5.0, domain_max=10.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(x * np.arange(start=1, stop=(d + 1), dtype=x.dtype) / 2, axis=-1)
        result = sum1 + sum2**2 + sum2**4

        return result


class BohachevskyNumpy(NumpyFunction):
    """Bohachevsky function (f1, 2-D) defined in [1]."""

    def __init__(self, dims: int, domain_min=-100.0, domain_max=100.0):
        assert dims == 2, "Bohachevsky only supports 2d."
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        result = (
            np.power(x[0], 2)
            + 2 * np.power(x[1], 2)
            - 0.3 * np.cos(3 * np.pi * x[0])
            - 0.4 * np.cos(4 * np.pi * x[1])
            + 0.7
        )

        return result


class SumSquaresNumpy(NumpyFunction):
    """SumSquares function defined in [1]."""

    def __init__(self, dims: int, domain_min=-10.0, domain_max=10.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        d = x.shape[-1]
        mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
        result = np.sum((x**2) * mul, axis=-1)

        return result


class SphereNumpy(NumpyFunction):
    """Sphere function defined in [1]."""

    def __init__(self, dims: int, domain_min=-5.12, domain_max=5.12):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        result = np.sum(x * x, axis=-1)

        return result


class RotatedHyperEllipsoidNumpy(NumpyFunction):
    """Rotated Hyper-Ellipsoid function defined in [1]."""

    def __init__(self, dims: int, domain_min=-65.536, domain_max=65.536):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        x = np.atleast_2d(x)
        mat = x[:, None].repeat(x.shape[-1], axis=1)
        matlow = np.tril(mat)
        inner = np.sum(matlow**2, axis=-1)
        result = np.sum(inner, axis=-1)

        return np.squeeze(result)


class DixonPriceNumpy(NumpyFunction):
    """Dixon-Price function defined in [1]."""

    def __init__(self, dims: int, domain_min=-10.0, domain_max=10.0):
        super().__init__(dims, domain_min=domain_min, domain_max=domain_max)

    def _fn(self, x: np.ndarray):
        x = np.atleast_2d(x)
        x0 = x[:, 0]
        d = x.shape[-1]
        ii = np.arange(2.0, d + 1)
        xi = x[:, 1:]
        xold = x[:, :-1]
        dixon_sum = ii * (2 * xi**2 - xold) ** 2
        result = (x0 - 1) ** 2 + np.sum(dixon_sum, axis=-1)

        return np.squeeze(result)
